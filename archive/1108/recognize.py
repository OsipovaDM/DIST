"""Распознавание ПДн: правила + BERT.

Секции:
    1. Валидаторы контрольных разрядов
    2. Регулярки для структурированных атрибутов
    3. Загрузка BERT (ленивая) и прогон
    4. predict() — объединение результатов
"""
import re
from typing import List, Tuple
from config import MODEL_OUT, LABELS, LABEL2ID, MAX_LEN

# ============================================================
# PASSPORT_REGION_WHITELIST — используется только здесь (PASSPORT и KLADR)
# ============================================================
PASSPORT_REGION_WHITELIST = {
    1,3,4,5,7,8,10,11,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,
    32,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
    54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,
    75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,
    96,97,98,99,
}

# ============================================================
# Секция 1. Валидаторы (копия ячейки 2 КМ2)
# ============================================================
# 2. Функции проверки контрольных разрядов
# ============================================================
def _digits_only(s: str) -> str:
    """Оставляет в строке только цифры — убирает пробелы, дефисы, точки, скобки."""
    return re.sub(r"\D", "", s)

def validate_inn(digits: str) -> bool:
    """Проверяет 12-значный ИНН физического лица по алгоритму ФНС."""

    # Две последние цифры ИНН — контрольные. Они вычисляются как взвешенная сумма
    # первых 10 и 11 цифр с фиксированными коэффициентами, взятая по модулю 11
    # и затем по модулю 10. Совпадение с фактическими цифрами означает корректный ИНН.

    d = _digits_only(digits)
    if len(d) != 12: return False
    nums = [int(c) for c in d]
    c11 = [7,2,4,10,3,5,9,4,6,8]    # коэффициенты первый контрольный разряд
    c12 = [3,7,2,4,10,3,5,9,4,6,8]  # коэффициенты второй контрольный разряд
    n11 = (sum(nums[i]*c11[i] for i in range(10)) % 11) % 10
    if n11 != nums[10]: return False
    n12 = (sum(nums[i]*c12[i] for i in range(11)) % 11) % 10
    return n12 == nums[11]

def validate_luhn(card_number: str) -> bool:
    """Проверяет 16-значный номер банковской карты по алгоритму Луна."""
    
    # каждая вторая цифра справа умножается на 2; если произведение > 9,
    # из него вычитается 9 (эквивалентно сумме цифр). Сумма всех цифр должна
    # делиться на 10 — это признак валидного номера.
    
    d = _digits_only(card_number)
    if len(d) != 16: return False
    total = 0
    for i, ch in enumerate(reversed(d)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9: n -= 9
        total += n
    return total % 10 == 0

def validate_passport_series(series: str) -> bool:
    """Проверяет серию паспорта РФ (4 цифры) по белому списку регионов."""
    
    # Первые две цифры серии — код субъекта РФ, выдавшего паспорт. Если код
    # не входит в перечень действующих регионов, не документ.
    
    s = _digits_only(series)
    if len(s) != 4: return False
    return int(s[:2]) in PASSPORT_REGION_WHITELIST

def validate_ogrnip(digits: str) -> bool:
    """Проверяет 15-значный ОГРНИП индивидуального предпринимателя."""
    
    # Контрольная цифра — это остаток от деления первых 14 цифр на 13, взятый
    # по модулю 10. Если совпадает с 15-й цифрой — номер корректный.
    
    d = _digits_only(digits)
    if len(d) != 15: return False
    return (int(d[:14]) % 13) % 10 == int(d[14])

def validate_kladr(digits: str) -> bool:
    """Проверяет код КЛАДР (13 или 17 цифр)."""
    
    # КЛАДР — структурированный код адреса: регион (2 цифры) + район (3) + город (3)
    # + населённый пункт (3) + улица (4) и т.д. Полноценная проверка требует сверки
    # всех уровней, но минимальная гарантия — регион должен быть реальным.
    
    d = _digits_only(digits)
    if len(d) not in (13, 17): return False
    return int(d[:2]) in PASSPORT_REGION_WHITELIST

assert validate_inn("770698959438") is True
assert validate_luhn("4561261212345467") is True
assert validate_passport_series("4506") is True
assert validate_passport_series("0000") is False
assert validate_ogrnip("304500116000157") is True
assert validate_kladr("7700000000000") is True
assert validate_kladr("0000000000000") is False
print("✅Контрольные функции OK.")

# ============================================================
# Секция 2. Регулярки (копия ячейки 6 КМ2)
# ============================================================
# Ячейка 6. Правила для структурированных атрибутов
# ============================================================

# E-mail: локальная часть + @ + домен + точка + зона (2+ символа).
# Флаг (?i) — регистронезависимо, чтобы Ivan@Mail.RU тоже находился.
EMAIL_RE = re.compile(
    r"(?i)([a-z0-9а-яё_.+\-]+@[a-zа-яё\d][a-zа-яё\d.\-]*\.[a-zа-яё]{2,})"
)

# IPv4: четыре числа 0–255 через точку. Lookbehind (?<![\d.]) защищает
# от захвата «хвоста» более длинного числа, lookahead (?!\.?\d) — от захвата
# начальной части, если после IP стоит ещё одна цифра через точку.
IP_RE = re.compile(
    r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)(?!\.?\d)"
)


def extract_ip(text):
    """Находит все IPv4-адреса в тексте. Контрольных разрядов у IP нет,
    проверка структурная — только формат четырёх октетов 0–255."""
    return [("IP", m.start(), m.end()) for m in IP_RE.finditer(text)]


# ИНН: 12 цифр, между ними допустимы пробелы и дефисы.
# Lookbehind/lookahead не дают захватить кусок более длинного числа.
INN_RE = re.compile(r"(?<!\d)(\d[\s\-]?){11}\d(?!\d)")


def extract_inn(text):
    """Находит 12-значные ИНН и валидирует контрольные разряды."""
    out = []
    for m in INN_RE.finditer(text):
        if validate_inn(re.sub(r"\D", "", m.group(0))):
            out.append(("INN", m.start(), m.end()))
    return out


# Банковская карта: 16 цифр с возможными пробелами/дефисами
CARD_RE = re.compile(r"(?<!\d)(?:\d[\s\-]?){15}\d(?!\d)")


def extract_card(text):
    """Находит 16-значные номера карт и валидирует по алгоритму Луна."""
    out = []
    for m in CARD_RE.finditer(text):
        if validate_luhn(re.sub(r"\D", "", m.group(0))):
            out.append(("CREDIT_CARD", m.start(), m.end()))
    return out


# Паспорт: 2+2 цифры серии и 6 цифр номера. Между группами допустимы
# пробел или дефис.
PASSPORT_RE = re.compile(r"(?<!\d)(\d{2})[\s\-]?(\d{2})[\s\-]?(\d{6})(?!\d)")


def extract_passport(text):
    """Находит паспортные данные и проверяет серию по белому списку регионов."""
    out = []
    for m in PASSPORT_RE.finditer(text):
        if validate_passport_series(m.group(1) + m.group(2)):
            out.append(("PASSPORT", m.start(), m.end()))
    return out


# ОГРНИП: ровно 15 цифр
OGRNIP_RE = re.compile(r"(?<!\d)(\d[\s\-]?){14}\d(?!\d)")


def extract_ogrnip(text):
    """Находит ОГРНИП и проверяет контрольную цифру по алгоритму ФНС."""
    out = []
    for m in OGRNIP_RE.finditer(text):
        if validate_ogrnip(re.sub(r"\D", "", m.group(0))):
            out.append(("OGRNIP", m.start(), m.end()))
    return out


# КЛАДР: 13 или 17 цифр подряд (без разделителей — так устроен этот код)
KLADR_RE = re.compile(r"(?<!\d)(\d{13}|\d{17})(?!\d)")


def extract_kladr(text):
    """Находит код КЛАДР и проверяет, что первые 2 цифры — валидный регион."""
    out = []
    for m in KLADR_RE.finditer(text):
        if validate_kladr(m.group(0)):
            out.append(("KLADR", m.start(), m.end()))
    return out


# Свидетельство о рождении: римская серия I–XV, опционально две буквы субъекта
# (АК, МЮ, МК и т.п.), затем 6–7 цифр номера. Между частями — пробел или дефис.
BIRTH_CERT_RE = re.compile(
    r"(?<![А-ЯЁA-Z\d])([IVX]{1,4})[\s\-]?(?:[А-ЯЁA-Z]{2}[\s\-]?)?(\d{6,7})(?!\d)"
)


def extract_birth_cert(text):
    """Находит номера свидетельств о рождении. Контрольного разряда нет,
    проверка только структурная (серия + номер)."""
    return [("BIRTH_CERT", m.start(), m.end()) for m in BIRTH_CERT_RE.finditer(text)]


def extract_by_rules(text):
    """Прогоняет все регулярки и возвращает объединённый список найденных сущностей."""
    
    # Возвращает список кортежей (тип атрибута, начало, конец) — в символьных
    # индексах исходного текста. Именно в таком виде работает predict().
    
    if not text:
        return []
    res = []
    res += [("EMAIL", m.start(), m.end()) for m in EMAIL_RE.finditer(text)]
    res += extract_ip(text)
    res += extract_inn(text)
    res += extract_card(text)
    res += extract_passport(text)
    res += extract_ogrnip(text)
    res += extract_kladr(text)
    res += extract_birth_cert(text)
    return res


# Демонстрация: все 8 структурированных атрибутов на одном тексте
demo = ("Иван, почта ivan@mail.ru, ИНН 770698959438, "
        "карта 4561 2612 1234 5467, паспорт 45 06 123456, "
        "ОГРНИП 304500116000157, IP 192.168.1.100, "
        "КЛАДР 7700000000000, свидетельство IV-МЮ 123456")
print("Правила на демо:")
for x in extract_by_rules(demo):
    print(f"  {x[0]:15s} '{demo[x[1]:x[2]]}'")

# ============================================================
# Секция 3. BERT (заменяет прямую загрузку из ячейки 7 КМ2)
# ============================================================
_ner_pipe = None
_tokenizer = None

def _get_pipeline():
    """Ленивая загрузка BERT: при `import pdn` модель НЕ грузится,
    только при первом вызове predict(). Это критично для скорости
    тестов и для возможности импортировать пакет без готовой модели."""
    global _ner_pipe, _tokenizer
    if _ner_pipe is None:
        from transformers import pipeline as hf_pipeline, AutoTokenizer
        try:
            _ner_pipe = hf_pipeline(
                "token-classification", model=MODEL_OUT, tokenizer=MODEL_OUT,
                aggregation_strategy="simple", device=-1,
            )
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_OUT)
        except Exception as e:
            print(f"⚠ BERT не загружен, работаем только по правилам: {e}")
            _ner_pipe = False            # sentinel: не пробовать снова
            _tokenizer = None
    return _ner_pipe, _tokenizer

def model_entities(text):
    """Прогоняет текст через BERT и возвращает список сущностей (SURNAME/NAME)."""
            
    # Текст режется на перекрывающиеся чанки по MAX_LEN токенов, потому что
    # документы factRuEval длиннее 512 токенов, а BERT видит только первые 512.
    # Без чанкинга половина ФИО в длинных документах не находилась бы.
    # Перекрытие 50% нужно, чтобы сущность на границе окна не разрывалась.
        
    ner_pipe, tokenizer = _get_pipeline()   # ← добавить
    if not text or ner_pipe is False or ner_pipe is None:   # ← заменить проверку
        return []
    
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    if not offsets:
        return []

    chunk_size = MAX_LEN - 4          # -4 под [CLS], [SEP] и запас
    stride = chunk_size // 2          # перекрытие 50%
    out = []
    seen = set()                      # дедупликация между чанками

    start_idx = 0
    while start_idx < len(offsets):
        end_idx = min(start_idx + chunk_size, len(offsets))
        char_start = offsets[start_idx][0]
        char_end = offsets[end_idx - 1][1]
        chunk_text = text[char_start:char_end]

        for ent in ner_pipe(chunk_text):
            label = ent["entity_group"]
            if label not in ("NAME", "SURNAME"):
                continue
            abs_start = char_start + int(ent["start"])
            abs_end = char_start + int(ent["end"])
            key = (label, abs_start, abs_end)
            if key in seen:
                continue
            seen.add(key)
            out.append((label, abs_start, abs_end))

        if end_idx >= len(offsets):
            break
        start_idx += stride

    return out
    
    
def _overlap(a, b):
    """Проверяет, пересекаются ли два спана по символьным позициям.
    Пересечение = не (один полностью до другого или полностью после)."""
    return not (a[2] <= b[1] or a[1] >= b[2])


def merge_entities(rule_ents, model_ents):
    """Объединяет результаты правил и модели по принципу «правило важнее»."""
    
    # 1. Сначала идут все сущности, найденные правилами.
    # 2. При конфликте (пересечение спанов) BERT-сущность отбрасывается —
    #    правила точнее на структурированных атрибутах.
    # 3. Если два спана одного атрибута пересекаются (редкий случай, когда
    #    правило и BERT нашли одно и то же), оставляется более длинный.
    
    all_ents = [(l, s, e, "rule") for l, s, e in rule_ents] + \
                [(l, s, e, "model") for l, s, e in model_ents]
    # Сортируем: сначала все rule, затем более длинные — они выиграют конфликт
    all_ents.sort(key=lambda x: (0 if x[3] == "rule" else 1, -(x[2] - x[1])))
    result, used = [], []
    for l, s, e, src in all_ents:
        conflict = any(_overlap((l, s, e), (ul, us, ue)) for ul, us, ue in used)
        if conflict:
            same = [(ul, us, ue) for ul, us, ue in used
                    if ul == l and _overlap((l, s, e), (ul, us, ue))]
            if same:
                longer = max(same, key=lambda x: x[2] - x[1])
                if (e - s) > (longer[2] - longer[1]):
                    used = [u for u in used if u != longer]
                    result = [r for r in result if r != longer]
                    result.append((l, s, e))
                    used.append((l, s, e))
            continue
        result.append((l, s, e))
        used.append((l, s, e))
    result.sort(key=lambda x: x[1])
    return result


def predict(text):
    """Финальная функция распознавания ПДн."""
    
    # Возвращает список кортежей (тип атрибута, начало, конец) в символьных
    # индексах исходного текста. Формат соответствует требованию задания.
    
    if not text or not text.strip():
        return []
    return merge_entities(extract_by_rules(text), model_entities(text))