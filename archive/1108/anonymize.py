"""Обезличивание ПДн: морфология + замена + сквозной пайплайн.

Секции:
    1. Морфология (pymorphy3)
    2. Обезличивание структурированных атрибутов
    3. anon() — главная функция
    4. recognize_and_anonymize() — полный цикл
"""
import re
import random
import pymorphy3
from faker import Faker

# Валидаторы и predict берём из соседнего модуля — не дублируем.
# В ячейке 1 ЛР2 они были локальными (validate_inn, validate_luhn) —
# здесь их НЕ надо копировать, они уже есть в recognize.py.
from typing import List, Tuple, Dict, Optional      # ← добавили Dict, Optional
from recognize import predict, validate_luhn         # ← убрали validate_inn, добавили точку
from config import OBFUSCATABLE                      # ← from config вместо from .

# Глобальные объекты (создаются один раз при импорте — это дёшево)
morph = pymorphy3.MorphAnalyzer()
fake = Faker("ru_RU")
rng = random.SystemRandom()

# ============================================================
# Секция 1. Морфология — копия ячейки 2 ЛР2 ЦЕЛИКОМ
# ============================================================
# pymorphy3 разбирает слово в список возможных интерпретаций (Parse).
# Для ФИО это принципиально: «Стоян» может разобраться как что угодно,
# кроме имени (в словаре его нет). Если склонять такой разбор, получаются
# прилагательные вроде «стоянный». Поэтому генерируем имя/фамилию и
# проверяем, что pymorphy видит её как Name/Surn — иначе пробуем ещё раз.

GRAMMEME_KEYS = ("gender", "number", "case")


def _find_parse_with_tag(word: str, tag_name: str):
    """Возвращает первый разбор pymorphy, в теге которого есть tag_name.
    
    Для имён это "Name", для фамилий "Surn", для отчеств "Patr".
    Если ни один разбор не содержит нужного тега — возвращает None
    (слово не является именем/фамилией в словаре pymorphy).
    """
    for p in morph.parse(word):
        if tag_name in p.tag:
            return p
    return None


def get_grammemes(word: str) -> Dict[str, Optional[str]]:
    """Возвращает род, число и падеж слова.
    
    Приоритет отдаётся разбору как имени/фамилии/отчеству — если такой
    есть, потому что разбор как прилагательного может дать неверный
    падеж (у неодушевлённых прилагательных мужского рода именительный
    и винительный совпадают). Если специального разбора нет — берём
    первый попавшийся (для не-ФИО этого достаточно).
    """
    parses = morph.parse(word)
    if not parses:
        return {"gender": "masc", "number": "sing", "case": "nomn"}
    # Ищем «человеческий» разбор
    chosen = parses[0]
    for p in parses:
        if any(t in p.tag for t in ("Name", "Surn", "Patr")):
            chosen = p
            break
    tag = chosen.tag
    return {
        "gender": tag.gender or "masc",
        "number": tag.number or "sing",
        "case":   tag.case   or "nomn",
    }


def inflect_to(word: str, grammemes: Dict[str, Optional[str]]) -> str:
    """Приводит слово к заданным граммемам (число, падеж).
    
    Род не входит в целевую форму — он фиксирован самим словом.
    Если исходное слово не склоняется (неизменяемое имя вроде «Джон»),
    inflect() может вернуть None или исходную форму — обрабатываем это
    на уровне вызывающего кода (generate_*_with_morphology).
    """
    target = {g for g in (grammemes.get("case"), grammemes.get("number")) if g}
    if not target:
        return word
    parses = morph.parse(word)
    if not parses:
        return word
    inflected = parses[0].inflect(target)
    return inflected.word if inflected else word


def _generate_inflected(original: str, tag_name: str) -> str:
    """Универсальная генерация: имя или фамилия с сохранением граммем.
    
    tag_name: "Name" для имён, "Surn" для фамилий.
    
    Логика:
    1. Определяем целевые граммемы по оригиналу (get_grammemes).
    2. Пробуем до 200 раз: генерируем случайное слово через Faker,
       проверяем, что оно разбирается как Name/Surn (иначе — пропускаем),
       склоняем к целевой форме, снова проверяем разбор.
    3. Если за 200 попыток валидного результата не нашлось (крайне
       маловероятно), возвращаем оригинал без изменений.
    """
    gr = get_grammemes(original)
    target = {g for g in (gr["case"], gr["number"]) if g}
    for _ in range(200):
        if tag_name == "Name":
            raw = fake.first_name_female() if gr["gender"] == "femn" else fake.first_name_male()
        else:
            raw = fake.last_name_female() if gr["gender"] == "femn" else fake.last_name_male()

        # 1) Слово из Faker должно быть именем/фамилией по мнению pymorphy
        parse = _find_parse_with_tag(raw, tag_name)
        if parse is None:
            continue
        # 2) Faker отдаёт слова в именительном падеже — убеждаемся в этом
        if parse.tag.case not in (None, "nomn"):
            continue
        # 3) Склоняем к целевой форме
        inflected = parse.inflect(target)
        if inflected is None:
            continue
        # 4) Результат должен снова разбираться как Name/Surn
        result_parse = _find_parse_with_tag(inflected.word, tag_name)
        if result_parse is None:
            continue
        # 5) И падеж результата должен совпасть с целевым
        if gr["case"] and result_parse.tag.case != gr["case"]:
            continue
        return inflected.word
    # Fallback: все 200 попыток неудачны — возвращаем оригинал
    return original


def generate_name_with_morphology(original: str) -> str:
    """Генерирует случайное имя, сохраняя род, число и падеж оригинала.
    
    Пример: «Ивана» (м.р., ед.ч., род.п.) → «Петра» (м.р., ед.ч., род.п.).
    """
    return _generate_inflected(original, "Name")


def generate_surname_with_morphology(original: str) -> str:
    """Генерирует случайную фамилию, сохраняя род, число и падеж оригинала.
    
    Пример: «Ивановой» (ж.р., ед.ч., род.п.) → «Петровой» (ж.р., ед.ч., род.п.).
    """
    return _generate_inflected(original, "Surn")


# ============================================================
# Секция 2. Обезличивание структурированных — копия ячейки 3 ЛР2 ЦЕЛИКОМ
# ============================================================

def obfuscate_email(email: str) -> str:
    """Заменяет локальную часть e-mail на случайную строку.
    
    Домен после «@» сохраняется — это косвенный признак (например, корпоративный
    домен может быть значим для аналитики). Если строка не похожа на e-mail
    (нет «@» или локальная часть пуста) — возвращается без изменений.
    """
    if not isinstance(email, str) or "@" not in email:
        return email
    local, domain = email.split("@", 1)
    if not local or not domain:
        return email
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    # Длину сохраняем — чтобы не ломать визуальное форматирование
    new_local = "".join(rng.choice(alphabet) for _ in range(max(len(local), 4)))
    return f"{new_local}@{domain}"


def obfuscate_inn(inn: str) -> str:
    """Обезличивает ИНН: сохраняет первые 4 цифры и валидные контрольные разряды.
    
    Первые 4 цифры — код налогового органа (косвенный признак региона).
    Остальные 6 цифр (позиции 5–10) генерируются случайно, а 11-я и 12-я
    пересчитываются по алгоритму ФНС, чтобы номер оставался валидным.
    Разделители (пробелы, дефисы) сохраняются на своих местах.
    """
    if not isinstance(inn, str):
        return inn
    digits = re.sub(r"\D", "", inn)
    if len(digits) != 12:
        return inn
    prefix = digits[:4]
    middle = "".join(str(rng.randint(0, 9)) for _ in range(6))
    nums = [int(c) for c in (prefix + middle)]
    c11 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
    c12 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
    n11 = (sum(nums[i] * c11[i] for i in range(10)) % 11) % 10
    nums11 = nums + [n11]
    n12 = (sum(nums11[i] * c12[i] for i in range(11)) % 11) % 10
    new_digits = prefix + middle + str(n11) + str(n12)
    # Возвращаем цифры на те же позиции, что были в исходной строке
    it = iter(new_digits)
    return "".join(next(it) if c.isdigit() else c for c in inn)


def obfuscate_phone(phone: str) -> str:
    """Обезличивает номер телефона: сохраняет первые 4 цифры и нецифровые символы.
    
    Первые 4 цифры — префикс + код оператора (например, «+7 (999)» → 7999).
    Все скобки, дефисы, пробелы, плюсы остаются на своих местах — чтобы
    формат был визуально тем же. Остальные цифры генерируются случайно.
    """
    if not isinstance(phone, str):
        return phone
    digit_positions = [i for i, c in enumerate(phone) if c.isdigit()]
    if len(digit_positions) < 5:
        return phone
    # Индексы первых 4 цифр — их сохраняем
    fixed = set(digit_positions[:4])
    result = list(phone)
    for pos in digit_positions:
        if pos not in fixed:
            result[pos] = str(rng.randint(0, 9))
    return "".join(result)


def obfuscate_credit_card(card: str) -> str:
    """Обезличивает номер карты: сохраняет первые 6 и последние 4 цифры.
    
    Первые 6 — BIN (банк-эмитент), последние 4 обычно показываются
    пользователю в интерфейсах. Средние 6 цифр (позиции 7–12) генерируются.
    Контрольная цифра (16-я) фиксирована — поэтому 12-ю цифру подбираем
    перебором так, чтобы номер проходил алгоритм Луна.
    Разделители (пробелы, дефисы) сохраняются.
    """
    if not isinstance(card, str):
        return card
    digits = re.sub(r"\D", "", card)
    if len(digits) != 16:
        return card
    prefix = digits[:6]                  # d1..d6
    suffix = digits[12:16]               # d13..d16, d16 — контрольная
    rng_free = "".join(str(rng.randint(0, 9)) for _ in range(5))  # d7..d11
    new_digits = None
    # Перебираем d12 от 0 до 9 — ровно один вариант сделает сумму Луна кратной 10
    for d12 in range(10):
        candidate = prefix + rng_free + str(d12) + suffix
        if validate_luhn(candidate):
            new_digits = candidate
            break
    if new_digits is None:
        return card
    it = iter(new_digits)
    return "".join(next(it) if c.isdigit() else c for c in card)

# ============================================================
# Секция 3. anon — копия ячейки 4 ЛР2 ЦЕЛИКОМ
# ============================================================

def _obfuscate_value(attr: str, value: str) -> str:
    """Диспетчеризация: выбирает функцию обезличивания по типу атрибута.
    
    Если тип не поддерживается — возвращает значение без изменений.
    Все исключения внутри конкретных функций перехватываются на уровне anon,
    поэтому здесь достаточно простого соответствия.
    """
    dispatch = {
        "SURNAME":     generate_surname_with_morphology,
        "NAME":        generate_name_with_morphology,
        "EMAIL":       obfuscate_email,
        "INN":         obfuscate_inn,
        "PHONE":       obfuscate_phone,
        "CREDIT_CARD": obfuscate_credit_card,
    }
    func = dispatch.get(attr)
    return func(value) if func else value


def anon(text: str, entities: List[Tuple[str, int, int]]) -> str:
    """Обезличивает ПДн в тексте по списку сущностей.
    
    Параметры:
        text     — исходный текст;
        entities — список (ATTR, start, end) в символьных индексах текста.
    
    Возвращает текст с заменами. Крайние случаи:
        * пустой текст или пустой список — возвращает текст как есть;
        * некорректные индексы (start < 0, end > len, start >= end) — сущность
          пропускается;
        * перекрывающиеся сущности — оставляется только первая (по start),
          последующие отбрасываются;
        * неизвестный тип атрибута — значение остаётся без замены.
    
    Замены применяются справа налево (через сортировку по start по убыванию),
    чтобы индексы предыдущих сущностей не сдвигались после вставки более
    длинных новых значений. Однако для читаемости проще собирать результат
    через срезы от начала — это безопасно, так как все исходные индексы
    относятся к оригинальному тексту, а не к промежуточному результату.
    """
    if not text or not entities:
        return text
    
    # 1. Фильтруем некорректные и перекрывающиеся сущности
    valid = []
    for e in entities:
        if not (isinstance(e, (tuple, list)) and len(e) == 3):
            continue
        attr, start, end = e
        if not (isinstance(attr, str) and isinstance(start, int) and isinstance(end, int)):
            continue
        if start < 0 or end > len(text) or start >= end:
            continue
        valid.append((attr, start, end))
    valid.sort(key=lambda x: x[1])
    
    # Удаляем перекрытия: сохраняем более раннюю, отбрасываем всё, что
    # начинается до конца уже принятой
    filtered = []
    last_end = 0
    for attr, start, end in valid:
        if start < last_end:
            continue
        filtered.append((attr, start, end))
        last_end = end
    
    # 2. Собираем результат через срезы (индексы — от оригинального текста)
    parts = []
    prev = 0
    for attr, start, end in filtered:
        parts.append(text[prev:start])
        original = text[start:end]
        try:
            replaced = _obfuscate_value(attr, original)
        except Exception as exc:
            # Любая проблема с конкретной заменой не должна ломать весь текст
            print(f"⚠ Ошибка при обезличивании {attr}='{original}': {exc}")
            replaced = original
        parts.append(replaced)
        prev = end
    parts.append(text[prev:])
    return "".join(parts)

# ============================================================
# Секция 4. Сквозной пайплайн — НОВАЯ функция
# ============================================================
def recognize_and_anonymize(text: str) -> str:
    """Полный цикл: predict() → фильтр OBFUSCATABLE → anon().

    Пример:
        >>> recognize_and_anonymize("Иванов Иван, ivan@mail.ru")
        'Петрова Мария, k3j9x7@mail.ru'
    """
    if not text or not text.strip():
        return text
    ents = [e for e in predict(text) if e[0] in OBFUSCATABLE]
    return anon(text, ents) if ents else text