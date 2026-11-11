"""Распознавание ПДн: правила + BERT"""
import re
from typing import List, Tuple

from config import (
    MODEL_OUT, MAX_LEN, BERT_DEVICE, AGGREGATION_STRATEGY, CHUNK_STRIDE_RATIO,
    PASSPORT_REGION_WHITELIST, INN_COEFF_C11, INN_COEFF_C12, digits_only,
)

# ========== ВАЛИДАТОРЫ КОНТРОЛЬНЫХ РАЗРЯДОВ ==========
def validate_inn(digits: str) -> bool:
    """Проверка 12-значного ИНН по алгоритму ФНС."""
    d = digits_only(digits)                              # оставляем только цифры
    if len(d) != 12:                                     # ИНН физлица всегда 12 цифр
        return False
    nums = [int(c) for c in d]                           # раскладываем в список цифр
    n11 = (sum(nums[i] * INN_COEFF_C11[i] for i in range(10)) % 11) % 10  # 11-я цифра
    if n11 != nums[10]:                                  # не совпало — невалидно
        return False
    n12 = (sum(nums[i] * INN_COEFF_C12[i] for i in range(11)) % 11) % 10  # 12-я цифра
    return n12 == nums[11]                               # обе совпали — валидно


def validate_luhn(card_number: str) -> bool:
    """Проверка 16-значного номера карты по алгоритму Луна."""
    d = digits_only(card_number)                         # только цифры
    if len(d) != 16:                                     # карта всегда 16 цифр
        return False
    total = 0                                            # сумма для проверки кратности 10
    for i, ch in enumerate(reversed(d)):                 # идём справа налево
        n = int(ch)
        if i % 2 == 1:                                   # каждая вторая цифра справа
            n *= 2
            if n > 9:                                    # двузначное → сумма цифр
                n -= 9
        total += n
    return total % 10 == 0                               # кратно 10 — валидно


def validate_passport_series(series: str) -> bool:
    """Проверка серии паспорта (4 цифры) по белому списку регионов."""
    s = digits_only(series)                              # только цифры
    if len(s) != 4:                                      # серия всегда 4 цифры
        return False
    return int(s[:2]) in PASSPORT_REGION_WHITELIST       # первые 2 цифры — код региона


def validate_ogrnip(digits: str) -> bool:
    """Проверка 15-значного ОГРНИП по алгоритму ФНС."""
    d = digits_only(digits)                              # только цифры
    if len(d) != 15:                                     # ОГРНИП всегда 15 цифр
        return False
    return (int(d[:14]) % 13) % 10 == int(d[14])         # контрольная — остаток % 13 % 10


def validate_kladr(digits: str) -> bool:
    """Проверка кода КЛАДР (13 или 17 цифр): первые 2 — валидный регион."""
    d = digits_only(digits)                              # только цифры
    if len(d) not in (13, 17):                           # допустимые длины КЛАДР
        return False
    return int(d[:2]) in PASSPORT_REGION_WHITELIST       # регион должен быть реальным


# ========== РЕГУЛЯРКИ И ФУНКЦИИ ИЗВЛЕЧЕНИЯ ==========
EMAIL_RE = re.compile(r"(?i)([a-z0-9а-яё_.+\-]+@[a-zа-яё\d][a-zа-яё\d.\-]*\.[a-zа-яё]{2,})")  # e-mail, (?i) — без учёта регистра

IP_RE = re.compile(
    r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}"  # три первых октета 0-255
    r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)(?!\.?\d)"             # четвёртый октет, не часть большего числа
)

INN_RE = re.compile(r"(?<!\d)(\d[\s\-]?){11}\d(?!\d)")           # 12 цифр с необязательными разделителями

CARD_RE = re.compile(r"(?<!\d)(?:\d[\s\-]?){15}\d(?!\d)")        # 16 цифр с необязательными разделителями

PASSPORT_RE = re.compile(r"(?<!\d)(\d{2})[\s\-]?(\d{2})[\s\-]?(\d{6})(?!\d)")  # серия 2+2, номер 6

OGRNIP_RE = re.compile(r"(?<!\d)(\d[\s\-]?){14}\d(?!\d)")        # 15 цифр

KLADR_RE = re.compile(r"(?<!\d)(\d{13}|\d{17})(?!\d)")           # 13 или 17 цифр подряд

BIRTH_CERT_RE = re.compile(
    r"(?<![А-ЯЁA-Z\d])([IVX]{1,4})[\s\-]?(?:[А-ЯЁA-Z]{2}[\s\-]?)?(\d{6,7})(?!\d)"  # римская серия + опц. субъект + номер
)


def extract_ip(text: str) -> List[Tuple[str, int, int]]:
    """Все IPv4-адреса в тексте (контрольных разрядов у IP нет)."""
    return [("IP", m.start(), m.end()) for m in IP_RE.finditer(text)]


def extract_inn(text: str) -> List[Tuple[str, int, int]]:
    """Все валидные ИНН в тексте."""
    out = []                                             # сюда складываем находки
    for m in INN_RE.finditer(text):
        if validate_inn(m.group(0)):                     # только прошедшие контроль
            out.append(("INN", m.start(), m.end()))
    return out


def extract_card(text: str) -> List[Tuple[str, int, int]]:
    """Все валидные номера карт (по Луну)."""
    out = []
    for m in CARD_RE.finditer(text):
        if validate_luhn(m.group(0)):                    # контроль по Луну
            out.append(("CREDIT_CARD", m.start(), m.end()))
    return out


def extract_passport(text: str) -> List[Tuple[str, int, int]]:
    """Все валидные номера паспортов."""
    out = []
    for m in PASSPORT_RE.finditer(text):
        if validate_passport_series(m.group(1) + m.group(2)):  # серия = группа1+группа2
            out.append(("PASSPORT", m.start(), m.end()))
    return out


def extract_ogrnip(text: str) -> List[Tuple[str, int, int]]:
    """Все валидные ОГРНИП."""
    out = []
    for m in OGRNIP_RE.finditer(text):
        if validate_ogrnip(m.group(0)):                  # контроль по алгоритму ФНС
            out.append(("OGRNIP", m.start(), m.end()))
    return out


def extract_kladr(text: str) -> List[Tuple[str, int, int]]:
    """Все валидные КЛАДР."""
    out = []
    for m in KLADR_RE.finditer(text):
        if validate_kladr(m.group(0)):                   # только с валидным регионом
            out.append(("KLADR", m.start(), m.end()))
    return out


def extract_birth_cert(text: str) -> List[Tuple[str, int, int]]:
    """Все свидетельства о рождении (без контрольного разряда)."""
    return [("BIRTH_CERT", m.start(), m.end()) for m in BIRTH_CERT_RE.finditer(text)]


def extract_by_rules(text: str) -> List[Tuple[str, int, int]]:
    """Прогон всех 8 регулярок, объединённый список (ATTR, start, end)."""
    if not text:                                         # пустой текст — нечего искать
        return []
    res = []                                             # аккумулируем находки
    res += [("EMAIL", m.start(), m.end()) for m in EMAIL_RE.finditer(text)]  # email — без валидатора
    res += extract_ip(text)                              # IP
    res += extract_inn(text)                             # ИНН
    res += extract_card(text)                            # карта
    res += extract_passport(text)                        # паспорт
    res += extract_ogrnip(text)                          # ОГРНИП
    res += extract_kladr(text)                           # КЛАДР
    res += extract_birth_cert(text)                      # свидетельство
    return res


# ========== ЗАГРУЗКА BERT ==========
_ner_pipe = None    # сюда положим pipeline после первой загрузки
_tokenizer = None   # сюда — токенизатор


def _get_pipeline():
    """Ленивая загрузка BERT: при import recognize модель НЕ грузится.

    Загрузка — только при первом predict(). Это критично для скорости тестов
    и возможности импортировать модуль без готовой модели.
    Возвращает (pipeline, tokenizer) или (False, None), если модель не загрузилась.
    """
    global _ner_pipe, _tokenizer                      # изменяем модульные переменные
    if _ner_pipe is None:                             # ещё не пытались грузить
        from transformers import pipeline as hf_pipeline, AutoTokenizer  # импорт здесь — чтобы не тянуть transformers при import recognize
        try:
            _ner_pipe = hf_pipeline(
                "token-classification", model=MODEL_OUT, tokenizer=MODEL_OUT,
                aggregation_strategy=AGGREGATION_STRATEGY,  # способ склейки BIO в спаны
                device=BERT_DEVICE,                          # -1 = CPU
            )
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_OUT)  # тот же токенизатор, что при обучении
        except Exception as e:
            print(f"⚠ BERT не загружен, работаем только по правилам: {e}")
            _ner_pipe = False                         # sentinel: больше не пробовать
            _tokenizer = None
    return _ner_pipe, _tokenizer                      # (pipeline, tokenizer) либо (False, None)


def model_entities(text: str) -> List[Tuple[str, int, int]]:
    """Прогон BERT по тексту с чанкингом длинных документов.

    Документы factRuEval бывают 600–800 токенов, а BERT видит только MAX_LEN.
    Режем на перекрывающиеся окна, прогоняем по каждому, склеиваем результаты
    по абсолютным смещениям. Перекрытие — чтобы сущность на границе не рвалась.
    """
    ner_pipe, tokenizer = _get_pipeline()             # ленивая загрузка
    if not text or ner_pipe is False or ner_pipe is None:  # пусто или BERT не загрузился
        return []
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)  # токенизация с offsets
    offsets = enc["offset_mapping"]                   # список (char_start, char_end) для каждого токена
    if not offsets:                                   # пустая токенизация
        return []
    chunk_size = MAX_LEN - 4                          # -4 под [CLS], [SEP] и запас
    stride = int(chunk_size * CHUNK_STRIDE_RATIO)     # шаг = доля от chunk_size (50% перекрытия)
    out = []                                          # найденные сущности
    seen = set()                                      # дедупликация между чанками
    start_idx = 0                                     # стартовый токен текущего чанка
    while start_idx < len(offsets):                   # пока не прошли весь текст
        end_idx = min(start_idx + chunk_size, len(offsets))  # конец чанка (не выходим за границы)
        char_start = offsets[start_idx][0]            # символьное начало чанка
        char_end = offsets[end_idx - 1][1]            # символьный конец чанка
        chunk_text = text[char_start:char_end]        # вырезаем кусок текста
        for ent in ner_pipe(chunk_text):              # прогон BERT по чанку
            label = ent["entity_group"]               # NAME или SURNAME
            if label not in ("NAME", "SURNAME"):      # остальные атрибуты — по правилам
                continue
            abs_start = char_start + int(ent["start"])  # абсолютное начало в исходном тексте
            abs_end = char_start + int(ent["end"])      # абсолютный конец
            key = (label, abs_start, abs_end)         # ключ для дедупликации
            if key in seen:                           # уже видели в другом чанке
                continue
            seen.add(key)
            out.append((label, abs_start, abs_end))
        if end_idx >= len(offsets):                   # дошли до конца
            break
        start_idx += stride                           # сдвигаем окно вперёд
    return out


# ========== СЛИЯНИЕ И PREDICT ==========
def _overlap(a, b) -> bool:
    """Пересекаются ли два спана (по символьным позициям)."""
    return not (a[2] <= b[1] or a[1] >= b[2])         # не (один до другого или после)


def merge_entities(rule_ents, model_ents):
    """Слияние правил и модели: правило важнее, при равенстве — длиннее."""
    all_ents = [(l, s, e, "rule") for l, s, e in rule_ents] + \
               [(l, s, e, "model") for l, s, e in model_ents]   # единый список с пометкой источника
    all_ents.sort(key=lambda x: (0 if x[3] == "rule" else 1, -(x[2] - x[1])))  # сначала rule, потом длинные
    result, used = [], []                             # результат и уже занятые спаны
    for l, s, e, src in all_ents:
        if any(_overlap((l, s, e), u) for u in used): # конфликт с уже принятой сущностью
            same = [u for u in used if u[0] == l and _overlap((l, s, e), u)]  # тот же тип?
            if same:                                  # один атрибут нашли и правило, и модель
                longer = max(same, key=lambda x: x[2] - x[1])  # берём более длинный
                if (e - s) > (longer[2] - longer[1]):  # новый длиннее — заменяем
                    used = [u for u in used if u != longer]
                    result = [r for r in result if r != longer]
                    result.append((l, s, e))
                    used.append((l, s, e))
            continue                                  # конфликт разных типов — правило победило
        result.append((l, s, e))                      # конфликта нет — добавляем
        used.append((l, s, e))
    result.sort(key=lambda x: x[1])                   # сортировка по началу для читаемости
    return result


def predict(text: str) -> List[Tuple[str, int, int]]:
    """Финальная функция распознавания ПДн.

    Возвращает список (ATTR, start, end) в символьных индексах исходного текста.
    Пустой текст → пустой список. BERT грузится только при первом вызове.
    """
    if not text or not text.strip():                  # пусто/пробелы — нечего искать
        return []
    return merge_entities(extract_by_rules(text), model_entities(text))  # правила + BERT → слияние