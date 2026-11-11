"""Обезличивание ПДн: морфология + замена + сквозной пайплайн"""
import random, re
from typing import List, Tuple, Dict, Optional

import pymorphy3
from faker import Faker

from config import (
    EMAIL_ALPHABET, EMAIL_MIN_LOCAL_LEN,
    INN_COEFF_C11, INN_COEFF_C12, digits_only,
)
from recognize import predict, validate_luhn

morph = pymorphy3.MorphAnalyzer()    # морфологический анализатор (загружается один раз)
fake = Faker("ru_RU")                # генератор русских имён и фамилий
rng = random.SystemRandom()          # криптостойкий ГПСЧ — не зависит от seed

# ========== МОРФОЛОГИЯ ==========
def get_grammemes(word: str) -> Dict[str, Optional[str]]:
    """Род, число и падеж слова (приоритет — разбору как ФИО)."""
    parses = morph.parse(word)                                 # все разборы слова
    if not parses:                                             # пустой список — слово неизвестно
        return {"gender": "masc", "number": "sing", "case": "nomn"}  # разумные умолчания
    chosen = parses[0]                                         # по умолчанию — первый разбор
    for p in parses:                                           # ищем разбор как ФИО
        if any(t in p.tag for t in ("Name", "Surn", "Patr")):  # имя/фамилия/отчество
            chosen = p                                         # предпочитаем его
            break
    tag = chosen.tag                                           # тег выбранного разбора
    return {
        "gender": tag.gender or "masc",                        # род или «мужской»
        "number": tag.number or "sing",                        # число или «единственное»
        "case":   tag.case   or "nomn",                        # падеж или «именительный»
    }


def inflect_to(word: str, grammemes: Dict[str, Optional[str]]) -> str:
    """Приводит слово к заданным числу и падежу (род не меняется)."""
    target = {g for g in (grammemes.get("case"), grammemes.get("number")) if g}  # множество целевых граммем
    if not target:                                             # нечего склонять
        return word
    parses = morph.parse(word)                                 # разборы слова
    if not parses:                                             # слово неизвестно
        return word
    inflected = parses[0].inflect(target)                      # пытаемся просклонять
    return inflected.word if inflected else word               # если не вышло — оригинал


def _find_parse_with_tag(word: str, tag_name: str):
    """Первый разбор pymorphy с тегом tag_name (Name/Surn/Patr)."""
    for p in morph.parse(word):
        if tag_name in p.tag:
            return p
    return None


def get_grammemes(word: str) -> Dict[str, Optional[str]]:
    """Род, число и падеж слова (приоритет — разбору как ФИО)."""
    parses = morph.parse(word)
    if not parses:
        return {"gender": "masc", "number": "sing", "case": "nomn"}
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


def _lemma(word: str, attr: str) -> str:
    """Нормальная форма для кэша однокоренных.

    Для SURNAME приводим к мужской форме им.п. — чтобы «Иванов», «Иванова»,
    «Ивановой» имели одну лемму «иванов». Для NAME — normal_form в роде
    оригинала (Иван → иван, Мария → мария).
    """
    parses = morph.parse(word)
    if not parses:
        return word.lower()
    chosen = None
    for p in parses:                                           # ищем разбор как ФИО
        if attr == "SURNAME" and "Surn" in p.tag:
            chosen = p
            break
        if attr == "NAME" and "Name" in p.tag:
            chosen = p
            break
    if chosen is None:                                         # не нашли — берём первый
        chosen = parses[0]
    if attr == "SURNAME":                                      # для фамилий — мужская база
        masc = chosen.inflect({"masc", "nomn"})
        if masc:
            return masc.word.lower()
    return chosen.normal_form.lower()


def _generate_normal_fio(attr: str, grammemes: Dict[str, Optional[str]]) -> str:
    """Генерирует новое слово в нормальной форме.

    Для SURNAME — всегда мужская форма им.п. (Петров). При замене склоняем
    к роду/падежу оригинала. Для NAME — в роде оригинала (Мария → Анна).
    """
    if attr == "SURNAME":
        return fake.last_name_male()
    if grammemes.get("gender") == "femn":
        return fake.first_name_female()
    return fake.first_name_male()


def _inflect_new(new_normal: str, original: str) -> str:
    """Склоняет новое слово к граммам оригинала (род/число/падеж)."""
    grammemes = get_grammemes(original)
    target = {g for g in (grammemes.get("gender"),
                          grammemes.get("number"),
                          grammemes.get("case")) if g}
    if not target:
        return new_normal.title()
    parses = morph.parse(new_normal)
    if not parses:
        return new_normal.title()
    inflected = parses[0].inflect(target)
    return inflected.word.title() if inflected else new_normal.title()

# ========== ОБЕЗЛИЧИВАНИЕ СТРУКТУРИРОВАННЫХ АТРИБУТОВ ==========
def obfuscate_email(email: str) -> str:
    """Меняет локальную часть e-mail, сохраняя домен после @."""
    if not isinstance(email, str) or "@" not in email:         # не e-mail — не трогаем
        return email
    local, domain = email.split("@", 1)                        # локальная часть и домен
    if not local or not domain:                                # пусто до или после @
        return email
    new_local = "".join(rng.choice(EMAIL_ALPHABET)             # случайные символы из алфавита
                        for _ in range(max(len(local), EMAIL_MIN_LOCAL_LEN)))  # не короче минимума
    return f"{new_local}@{domain}"                             # склеиваем обратно


def obfuscate_inn(inn: str) -> str:
    """Меняет средние 6 цифр ИНН, сохраняя первые 4 и валидный контроль.

    Префикс (4 цифры) — код налогового органа, его не трогаем.
    Позиции 5–10 — случайные. Позиции 11–12 пересчитываются по ФНС,
    чтобы ИНН остался валидным. Разделители остаются на своих местах.
    """
    if not isinstance(inn, str):                               # не строка — не трогаем
        return inn
    digits = digits_only(inn)                                  # только цифры
    if len(digits) != 12:                                      # не ИНН — не трогаем
        return inn
    prefix = digits[:4]                                        # код налогового органа
    middle = "".join(str(rng.randint(0, 9)) for _ in range(6)) # случайные 6 цифр
    nums = [int(c) for c in (prefix + middle)]                 # первые 10 цифр числом
    n11 = (sum(nums[i] * INN_COEFF_C11[i] for i in range(10)) % 11) % 10  # 11-я цифра
    nums11 = nums + [n11]                                      # 11 цифр для расчёта 12-й
    n12 = (sum(nums11[i] * INN_COEFF_C12[i] for i in range(11)) % 11) % 10  # 12-я цифра
    new_digits = prefix + middle + str(n11) + str(n12)         # готовый новый ИНН
    it = iter(new_digits)                                      # итератор по новым цифрам
    return "".join(next(it) if c.isdigit() else c for c in inn)  # вставляем на места цифр


def obfuscate_credit_card(card: str) -> str:
    """Меняет средние 6 цифр карты, сохраняя BIN, последние 4 и валидный Луна.

    Первые 6 — BIN эмитента. Последние 4 — обычно видимая часть номера.
    Позиции 7–11 — случайные. Позиция 12 подбирается перебором так,
    чтобы сумма по Луну делилась на 10 (карта осталась валидной).
    """
    if not isinstance(card, str):                              # не строка — не трогаем
        return card
    digits = digits_only(card)                                 # только цифры
    if len(digits) != 16:                                      # не карта — не трогаем
        return card
    prefix = digits[:6]                                        # BIN (сохраняем)
    suffix = digits[12:16]                                     # последние 4 (сохраняем)
    rng_free = "".join(str(rng.randint(0, 9)) for _ in range(5))  # d7..d11 — случайные
    new_digits = None                                          # сюда положим валидный вариант
    for d12 in range(10):                                      # перебор d12 — ровно один даст валидный Луна
        candidate = prefix + rng_free + str(d12) + suffix      # пробный номер
        if validate_luhn(candidate):                           # прошёл по Луну?
            new_digits = candidate                             # да — запоминаем
            break
    if new_digits is None:                                     # не должно случиться, но защищаемся
        return card
    it = iter(new_digits)                                      # итератор по новым цифрам
    return "".join(next(it) if c.isdigit() else c for c in card)  # вставляем на места цифр


def obfuscate_passport(passport: str) -> str:
    """Серия (4 цифры) сохраняется, номер (6 цифр) меняется.

    Косвенный признак — серия паспорта (код региона + код подразделения)
    остаётся прежней, меняется только номер. Разделители сохраняются.
    """
    if not isinstance(passport, str):
        return passport
    digits = digits_only(passport)
    if len(digits) != 10:                                    # не паспорт — не трогаем
        return passport
    new_number = "".join(str(rng.randint(0, 9)) for _ in range(6))  # новые 6 цифр
    new_digits = digits[:4] + new_number                     # серия + новый номер
    it = iter(new_digits)                                    # вставляем на места цифр
    return "".join(next(it) if c.isdigit() else c for c in passport)


def obfuscate_ogrnip(ogrnip: str) -> str:
    """Первые 4 цифры сохраняются, остальные 10 меняются, контроль пересчитывается.

    Первые 2 цифры — признак и год регистрации ИП, ещё 2 — код региона.
    Позиции 5–14 — случайные. Контрольная (15-я) считается по ФНС,
    чтобы ОГРНИП остался валидным.
    """
    if not isinstance(ogrnip, str):
        return ogrnip
    digits = digits_only(ogrnip)
    if len(digits) != 15:                                    # не ОГРНИП — не трогаем
        return ogrnip
    prefix = digits[:4]                                      # сохраняем 4 первые
    middle = "".join(str(rng.randint(0, 9)) for _ in range(10))  # 10 случайных
    first_14 = prefix + middle                               # 14 цифр для контроля
    ctrl = (int(first_14) % 13) % 10                         # контрольная по ФНС
    new_digits = first_14 + str(ctrl)
    it = iter(new_digits)
    return "".join(next(it) if c.isdigit() else c for c in ogrnip)


def obfuscate_ip(ip: str) -> str:
    """Первые 2 октета (сеть) сохраняются, последние 2 (хост) меняются.

    Сохраняем префикс сети — это косвенный признак провайдера/региона.
    Меняем только хостовую часть.
    """
    if not isinstance(ip, str):
        return ip
    parts = ip.split(".")
    if len(parts) != 4:                                      # не IPv4 — не трогаем
        return ip
    try:
        int(parts[0]); int(parts[1])                         # проверяем что это числа
    except ValueError:
        return ip
    new_p3 = rng.randint(1, 254)                             # новый 3-й октет
    new_p4 = rng.randint(1, 254)                             # новый 4-й октет
    return f"{parts[0]}.{parts[1]}.{new_p3}.{new_p4}"


def obfuscate_birth_cert(cert: str) -> str:
    """Серия (римская) и код субъекта сохраняются, номер (6-7 цифр) меняется.

    Косвенные признаки — римская серия (I–XV) и код субъекта (АК, МЮ, …)
    остаются. Меняется только числовой номер в конце.
    """
    if not isinstance(cert, str):
        return cert
    matches = list(re.finditer(r"\d{6,7}", cert))            # ищем группы 6-7 цифр
    if not matches:                                          # не свидетельство — не трогаем
        return cert
    m = matches[-1]                                          # последняя группа — номер
    new_num = "".join(str(rng.randint(0, 9)) for _ in range(len(m.group(0))))
    return cert[:m.start()] + new_num + cert[m.end():]       # серия остаётся нетронутой


def obfuscate_kladr(kladr: str) -> str:
    """Код региона (первые 2 цифры) сохраняется, остальные цифры меняются.

    Косвенный признак — регион. Остальные уровни (район, город, улица)
    генерируются случайно. КЛАДР пишется без разделителей, поэтому
    просто склеиваем регион + случайные цифры.
    """
    if not isinstance(kladr, str):
        return kladr
    digits = digits_only(kladr)
    if len(digits) not in (13, 17):                          # не КЛАДР — не трогаем
        return kladr
    region = digits[:2]                                      # код региона сохраняем
    rest = "".join(str(rng.randint(0, 9)) for _ in range(len(digits) - 2))
    return region + rest


# ========== ГЛАВНАЯ ФУНКЦИЯ anon() ==========
def _obfuscate_value(attr: str, value: str) -> str:
    """Диспетчер для структурированных атрибутов (ФИО — отдельно в anon)."""
    dispatch = {
        "EMAIL":       obfuscate_email,
        "INN":         obfuscate_inn,
        "CREDIT_CARD": obfuscate_credit_card,
        "PASSPORT":    obfuscate_passport,
        "OGRNIP":      obfuscate_ogrnip,
        "IP":          obfuscate_ip,
        "BIRTH_CERT":  obfuscate_birth_cert,
        "KLADR":       obfuscate_kladr,
    }
    func = dispatch.get(attr)
    return func(value) if func else value


def anon(text: str, entities: List[Tuple[str, int, int]]) -> str:
    """Обезличивает ПДн с идемпотентностью и однокоренными заменами.

    Идемпотентность: одинаковые сущности → одинаковые замены (кэш).
    Однокоренные: «Иванов», «Иванова», «Ивановой» → «Петров», «Петрова»,
    «Петровой» (одна лемма → одно новое слово, разные падежные формы).
    Кэш живёт в рамках одного вызова — между вызовами значения разные.
    """
    if not text or not entities:
        return text

    # --- Фильтрация валидных сущностей ---
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

    # --- Сортировка и удаление перекрытий ---
    valid.sort(key=lambda x: x[1])
    filtered = []
    last_end = 0
    for attr, start, end in valid:
        if start < last_end:
            continue
        filtered.append((attr, start, end))
        last_end = end

    # --- Кэш замен (идемпотентность + однокоренные) ---
    cache: Dict[Tuple[str, str], str] = {}
    parts = []
    prev = 0
    for attr, start, end in filtered:
        parts.append(text[prev:start])
        original = text[start:end]
        try:
            if attr in ("SURNAME", "NAME"):
                # ключ — по лемме: «Иванов» и «Иванова» → одна замена
                lemma = _lemma(original, attr)
                key = (attr, lemma)
                if key not in cache:
                    cache[key] = _generate_normal_fio(attr, get_grammemes(original))
                replaced = _inflect_new(cache[key], original)   # склоняем к падежу оригинала
            else:
                # ключ — по точному значению: два одинаковых ИНН → один новый
                key = (attr, original)
                if key not in cache:
                    cache[key] = _obfuscate_value(attr, original)
                replaced = cache[key]
        except Exception as exc:
            print(f"⚠ Ошибка при обезличивании {attr}='{original}': {exc}")
            replaced = original
        parts.append(replaced)
        prev = end
    parts.append(text[prev:])
    return "".join(parts)


# ========== СКВОЗНОЙ ПАЙПЛАЙН ==========
def recognize_and_anonymize(text: str) -> str:
    """Полный цикл: predict() → anon(). Обезличиваются все распознанные атрибуты.

    Список поддерживаемых атрибутов = ATTRS. Отдельного OBFUSCATABLE
    больше нет — все, что распознано, обезличивается.
    """
    if not text or not text.strip():
        return text
    ents = predict(text)                                     # все распознанные сущности
    return anon(text, ents) if ents else text