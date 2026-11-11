"""Обезличивание ПДн: морфология + замена + сквозной пайплайн"""
import random
from typing import List, Tuple, Dict, Optional

import pymorphy3
from faker import Faker

from config import (
    OBFUSCATABLE, EMAIL_ALPHABET, EMAIL_MIN_LOCAL_LEN, MAX_NAME_GEN_ATTEMPTS,
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


def obfuscate_phone(phone: str) -> str:
    """Меняет все цифры кроме первых 4, сохраняя скобки, дефисы, пробелы."""
    if not isinstance(phone, str):                             # не строка — не трогаем
        return phone
    digit_positions = [i for i, c in enumerate(phone) if c.isdigit()]  # позиции всех цифр
    if len(digit_positions) < 5:                               # слишком короткий номер
        return phone
    fixed = set(digit_positions[:4])                           # индексы первых 4 цифр (сохраняем)
    result = list(phone)                                       # копия строки как список символов
    for pos in digit_positions:                                # проходим по всем цифрам
        if pos not in fixed:                                   # не в префиксе — заменяем
            result[pos] = str(rng.randint(0, 9))
    return "".join(result)                                     # обратно в строку


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


# ========== ГЛАВНАЯ ФУНКЦИЯ anon() ==========
def _obfuscate_value(attr: str, value: str) -> str:
    """Диспетчер: вызывает нужный обфускатор по типу атрибута."""
    dispatch = {                                               # таблица «тип → функция»
        "SURNAME":     generate_surname_with_morphology,
        "NAME":        generate_name_with_morphology,
        "EMAIL":       obfuscate_email,
        "INN":         obfuscate_inn,
        "PHONE":       obfuscate_phone,
        "CREDIT_CARD": obfuscate_credit_card,
    }
    func = dispatch.get(attr)                                  # ищем функцию для типа
    return func(value) if func else value                      # нет функции — оставляем как есть


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
    """Полный цикл: predict() → фильтр OBFUSCATABLE → anon().

    Пример: «Иванов Иван, ivan@mail.ru» → «Петрова Мария, k3j9x7@mail.ru».
    Атрибуты, не входящие в OBFUSCATABLE (паспорт, ОГРНИП и т.п.),
    остаются в тексте нетронутыми.
    """
    if not text or not text.strip():                           # пусто — возвращаем как есть
        return text
    ents = [e for e in predict(text) if e[0] in OBFUSCATABLE]  # только обезличиваемые
    return anon(text, ents) if ents else text                  # замена или оригинал, если нечего