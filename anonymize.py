"""Обезличивание ПДн: морфология + замена + сквозной пайплайн"""
import random, re
from typing import List, Tuple, Dict, Optional

import pymorphy3
from faker import Faker

from config import (
    EMAIL_ALPHABET, EMAIL_MIN_LOCAL_LEN, MAX_NAME_GEN_ATTEMPTS,
    INN_COEFF_C11, INN_COEFF_C12, digits_only,
)
from recognize import predict, validate_luhn

morph = pymorphy3.MorphAnalyzer()    # морфологический анализатор (загружается один раз)
fake = Faker("ru_RU")                # генератор русских имён и фамилий
rng = random.SystemRandom()          # криптостойкий ГПСЧ — не зависит от seed

# ========== МОРФОЛОГИЯ ==========
def _find_parse_with_tag(word: str, tag_name: str):
    """Возвращает первый разбор pymorphy с тегом tag_name (Name/Surn/Patr).

    Нужно потому что «Стоян» может разобраться как прилагательное, и склонение
    даст «стоянный». Ищем именно человеческий разбор, иначе None.
    """
    for p in morph.parse(word):                                # перебираем все разборы
        if tag_name in p.tag:                                  # нашли нужный тег
            return p
    return None                                                # ни один разбор не подошёл


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


def _generate_inflected(original: str, tag_name: str) -> str:
    """Генерирует имя или фамилию с сохранением рода, числа, падежа.

    tag_name: «Name» — для имён, «Surn» — для фамилий.
    Пробует до MAX_NAME_GEN_ATTEMPTS раз: Faker даёт слово в именительном,
    проверяем что pymorphy видит его как ФИО, склоняем к целевой форме,
    проверяем что результат снова разбирается как ФИО и с нужным падежом.
    Если все попытки неудачны — возвращает оригинал.
    """
    gr = get_grammemes(original)                               # граммемы оригинала
    target = {g for g in (gr["case"], gr["number"]) if g}      # целевые число и падеж
    for _ in range(MAX_NAME_GEN_ATTEMPTS):                     # ограниченное число попыток
        if tag_name == "Name":                                 # генерируем имя
            raw = fake.first_name_female() if gr["gender"] == "femn" else fake.first_name_male()
        else:                                                  # генерируем фамилию
            raw = fake.last_name_female() if gr["gender"] == "femn" else fake.last_name_male()
        parse = _find_parse_with_tag(raw, tag_name)            # Faker дал слово нужного типа?
        if parse is None:
            continue                                           # не подошло — следующая попытка
        if parse.tag.case not in (None, "nomn"):               # Faker должен давать именительный
            continue
        inflected = parse.inflect(target)                      # склоняем к целевой форме
        if inflected is None:                                  # не склоняется — пропускаем
            continue
        result_parse = _find_parse_with_tag(inflected.word, tag_name)  # результат снова ФИО?
        if result_parse is None:
            continue
        if gr["case"] and result_parse.tag.case != gr["case"]: # падеж результата совпадает?
            continue
        return inflected.word                                  # всё ок — возвращаем
    return original                                            # 200 попыток не помогли — оригинал


def generate_name_with_morphology(original: str) -> str:
    """Случайное имя с сохранением рода, числа и падежа («Ивана» → «Петра»)."""
    return _generate_inflected(original, "Name")


def generate_surname_with_morphology(original: str) -> str:
    """Случайная фамилия с сохранением рода, числа и падежа («Ивановой» → «Петровой»)."""
    return _generate_inflected(original, "Surn")


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
    """Диспетчер: для каждого атрибута из ATTRS — своя функция обезличивания."""
    dispatch = {
        "SURNAME":     generate_surname_with_morphology,   # морфология
        "NAME":        generate_name_with_morphology,      # морфология
        "EMAIL":       obfuscate_email,                    # домен
        "INN":         obfuscate_inn,                      # 4 + контроль
        "CREDIT_CARD": obfuscate_credit_card,              # 6+4 + Луна
        "PASSPORT":    obfuscate_passport,                 # серия
        "OGRNIP":      obfuscate_ogrnip,                   # 4 + контроль
        "IP":          obfuscate_ip,                       # сеть
        "BIRTH_CERT":  obfuscate_birth_cert,               # серия
        "KLADR":       obfuscate_kladr,                    # регион
    }
    func = dispatch.get(attr)
    return func(value) if func else value


def anon(text: str, entities: List[Tuple[str, int, int]]) -> str:
    """Обезличивает ПДн в тексте по списку сущностей.

    text     — исходный текст;
    entities — список (ATTR, start, end) в символьных индексах.

    Крайние случаи:
      * пустой текст/список — возвращает текст как есть;
      * некорректные индексы — сущность пропускается;
      * перекрывающиеся сущности — оставляется первая (по start);
      * неизвестный тип атрибута — значение остаётся без замены;
      * исключение внутри обфускатора — логируется, значение не меняется.
    """
    if not text or not entities:                               # нечего обезличивать
        return text
    valid = []                                                 # прошедшие фильтр сущности
    for e in entities:                                         # проверяем каждую сущность
        if not (isinstance(e, (tuple, list)) and len(e) == 3): # должен быть кортеж из 3 элементов
            continue
        attr, start, end = e                                   # распаковка
        if not (isinstance(attr, str) and isinstance(start, int) and isinstance(end, int)):  # типы верные?
            continue
        if start < 0 or end > len(text) or start >= end:       # индексы в пределах текста?
            continue
        valid.append((attr, start, end))
    valid.sort(key=lambda x: x[1])                             # сортировка по началу
    filtered = []                                              # без перекрытий
    last_end = 0                                               # конец последней принятой
    for attr, start, end in valid:
        if start < last_end:                                   # начинается до конца предыдущей
            continue                                           # перекрытие — пропускаем
        filtered.append((attr, start, end))
        last_end = end
    parts = []                                                 # куски нового текста
    prev = 0                                                   # текущая позиция в оригинале
    for attr, start, end in filtered:
        parts.append(text[prev:start])                         # кусок до сущности
        original = text[start:end]                             # оригинальное значение
        try:
            replaced = _obfuscate_value(attr, original)        # обезличиваем
        except Exception as exc:                               # ошибка внутри обфускатора
            print(f"⚠ Ошибка при обезличивании {attr}='{original}': {exc}")  # логируем
            replaced = original                                # оставляем как есть
        parts.append(replaced)                                 # вставляем замену
        prev = end                                             # сдвигаем курсор
    parts.append(text[prev:])                                  # хвост после последней сущности
    return "".join(parts)                                      # склеиваем в строку


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