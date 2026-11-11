"""Оценка качества распознавания ПДн"""
import random
import re
from typing import List, Tuple

from faker import Faker

from config import (
    ATTRS, N_PER_ATTR, PASSPORT_REGION_WHITELIST,
    digits_only,
)
from recognize import (
    predict,
    validate_inn, validate_luhn, validate_passport_series,
    validate_ogrnip, validate_kladr,
)

fake_ru = Faker("ru_RU")   # генератор русских имён/фамилий/email

# ========== ШАБЛОНЫ И ГЕНЕРАТОРЫ ==========
MANUAL_TEMPLATES = {
    "SURNAME":     ["{SURNAME} пришёл на встречу.", "Мы позвонили {SURNAME_OBL}."],  # + винительный
    "NAME":        ["{NAME} написал письмо.",       "Передайте привет {NAME}."],     # именительный
    "EMAIL":       ["Пишите на {EMAIL} до пятницы.",   "Резервный адрес: {EMAIL}."],
    "INN":         ["ИНН компании: {INN}.",            "Укажите ИНН {INN} в декларации."],
    "CREDIT_CARD": ["Оплата на карту {CARD}.",         "Реквизиты: {CARD}."],
    "PASSPORT":    ["Паспорт: {PASSPORT}.",            "Серия и номер: {PASSPORT}."],
    "OGRNIP":      ["ОГРНИП: {OGRNIP}.",               "Регистрация ИП: {OGRNIP}."],
    "IP":          ["IP-адрес сервера: {IP}.",         "Соединение с {IP}."],
    "BIRTH_CERT":  ["Свидетельство о рождении: {BIRTH_CERT}.", "Выдан документ {BIRTH_CERT}."],
    "KLADR":       ["Код КЛАДР: {KLADR}.",             "Адрес по КЛАДР: {KLADR}."],
}


def gen_inn() -> str:
    """Случайный валидный 12-значный ИНН (контроль по алгоритму ФНС)."""
    while True:                                                # пока не получим валидный
        base = [random.randint(0, 9) for _ in range(10)]       # 10 случайных цифр
        c11 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]                  # коэффициенты 11-й
        n11 = (sum(base[i] * c11[i] for i in range(10)) % 11) % 10
        d11 = base + [n11]                                     # 11 цифр
        c12 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]               # коэффициенты 12-й
        n12 = (sum(d11[i] * c12[i] for i in range(11)) % 11) % 10
        res = "".join(map(str, base + [n11, n12]))             # готовый ИНН
        if validate_inn(res):                                  # проверка
            return res


def gen_card() -> str:
    """Случайный валидный 16-значный номер карты (алгоритм Луна)."""
    while True:
        body = "4" + "".join(str(random.randint(0, 9)) for _ in range(14))  # Visa, 15 цифр
        for cd in range(10):                                   # подбираем 16-ю
            cand = body + str(cd)
            if validate_luhn(cand):
                return cand


def gen_passport() -> str:
    """Паспорт с валидной серией; формат случайно из трёх."""
    region = random.choice(sorted(PASSPORT_REGION_WHITELIST))  # валидный код региона
    dept = random.randint(0, 99)                               # код подразделения
    series = f"{region:02d}{dept:02d}"                         # серия 4 цифры
    number = "".join(str(random.randint(0, 9)) for _ in range(6))  # номер 6 цифр
    fmt = random.choice(["space", "dash", "compact"])          # один из трёх форматов
    if fmt == "space":
        return f"{series[:2]} {series[2:]} {number}"           # «45 06 123456»
    if fmt == "dash":
        return f"{series[:2]}-{series[2:]}-{number}"           # «45-06-123456»
    return series + number                                     # «4506123456»


def gen_ogrnip() -> str:
    """Случайный валидный ОГРНИП (15 цифр, контроль по ФНС)."""
    while True:
        body = "3" + "".join(str(random.randint(0, 9)) for _ in range(13))  # первая цифра 3 (ИП)
        ctrl = (int(body) % 13) % 10                           # контрольная цифра
        res = body + str(ctrl)
        if validate_ogrnip(res):
            return res


def gen_ip() -> str:
    """Случайный IPv4: 4 октета 1–254 (без 0 и 255 — служебные)."""
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def gen_birth_cert() -> str:
    """Свидетельство о рождении в одном из трёх форматов."""
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
             "XI", "XII", "XIII", "XIV", "XV"]                 # римские серии
    series = random.choice(roman)                              # серия
    num = "".join(str(random.randint(0, 9)) for _ in range(6)) # номер
    fmt = random.choice(["plain", "subj_space", "subj_dash"])  # формат
    subj = random.choice(["АК", "МЮ", "МК", "АБ", "БА", "ЮК"]) # код субъекта
    if fmt == "plain":
        return f"{series} {num}"                               # «IV 123456»
    if fmt == "subj_space":
        return f"{series}-{subj} {num}"                        # «IV-МЮ 123456»
    return f"{series}-{subj}-{num}"                            # «IV-МЮ-123456»


def gen_kladr() -> str:
    """КЛАДР (13 цифр) с валидным кодом региона в начале."""
    region = random.choice(sorted(PASSPORT_REGION_WHITELIST))  # валидный регион
    rest = "".join(str(random.randint(0, 9)) for _ in range(11))  # остальные 11 цифр
    return f"{region:02d}{rest}"


def gen_email() -> str:
    """E-mail через Faker с ASCII-доменом — регулярка точно поймает."""
    return fake_ru.ascii_free_email()


def gen_name() -> str:
    """Случайное мужское имя из Faker ru_RU."""
    return fake_ru.first_name_male()


def gen_surname() -> str:
    """Случайная мужская фамилия из Faker ru_RU."""
    return fake_ru.last_name_male()


# ========== ПОДСЧЁТ tp/fp/fn И МЕТРИК ==========
def compute_per_attr(predict_fn, samples) -> dict:
    """Считает tp/fp/fn для каждого атрибута по спанам.

    samples — список (text, gold_spans). Спан засчитывается как tp,
    если есть предсказание того же типа с частичным перекрытием.
    Это мягче точного совпадения границ, но осмысленно для NER.
    """
    per_attr = {a: {"tp": 0, "fp": 0, "fn": 0} for a in ATTRS}  # счётчики по каждому атрибуту
    for text, gold in samples:                                  # по каждому примеру
        pset = set(predict_fn(text))                            # предсказания модели
        gset = set(gold)                                        # gold-спаны
        for a, s, e in gset:                                    # для каждого gold-спана
            if any(pa == a and not (e <= ps or s >= pe) for pa, ps, pe in pset):  # есть предсказание с перекрытием
                per_attr[a]["tp"] += 1
            else:
                per_attr[a]["fn"] += 1
        for a, s, e in pset:                                    # для каждого предсказания
            if not any(ga == a and not (e <= gs or s >= ge) for ga, gs, ge in gset):  # нет gold-спана
                per_attr[a]["fp"] += 1
    return per_attr


def metrics_from_per_attr(per_attr) -> list:
    """Считает precision, recall, F1 по tp/fp/fn для каждого атрибута."""
    rows = []
    for a in ATTRS:
        tp, fp, fn = per_attr[a]["tp"], per_attr[a]["fp"], per_attr[a]["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0             # precision
        rec = tp / (tp + fn) if (tp + fn) else 0.0              # recall
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0  # F1
        rows.append((a, prec, rec, f1, tp, fp, fn))             # строка для таблицы
    return rows


def _find_span(text: str, value: str) -> Tuple[int, int]:
    """(start, end) вхождения value в text, либо None."""
    idx = text.find(value)                                      # позиция подстроки
    return (idx, idx + len(value)) if idx >= 0 else None        # кортеж или None


# ========== ГЛАВНАЯ ФУНКЦИЯ ОЦЕНКИ ==========
def evaluate(factrueval_test_docs, n_per_attr: int = N_PER_ATTR) -> dict:
    """Оценивает predict() на Faker-тесте и factRuEval-тесте.

    factrueval_test_docs — [(text, gold_spans)] из dataset.load_factrueval();
    n_per_attr          — текстов на каждый атрибут (по умолчанию из config).

    Возвращает dict:
      rows        — список (attr, prec, rec, f1, tp, fp, fn);
      weighted_f1 — взвешенное по числу истинных примеров среднее F1;
      mean_f1     — простое среднее F1 по атрибутам;
      valid_data  — True, если все сгенерированные значения прошли контроль;
      cnt_faker   — сколько текстов на каждый атрибут;
      n_fact      — сколько документов в factRuEval-тесте.

    BERT грузится лениво внутри predict() при первом вызове.
    """
    # --- Генерируем Faker-тесты ---
    test_texts = []                                             # (text, attr, value) — value нужен для gold-спана
    for _ in range(n_per_attr):
        n, s = gen_name(), gen_surname()                        # пара имя-фамилия
        test_texts.append((random.choice(MANUAL_TEMPLATES["SURNAME"]).format(
            SURNAME=s, SURNAME_OBL=s + "у"), "SURNAME", s))
        test_texts.append((random.choice(MANUAL_TEMPLATES["NAME"]).format(
            NAME=n), "NAME", n))
        em = gen_email()
        test_texts.append((random.choice(MANUAL_TEMPLATES["EMAIL"]).format(EMAIL=em), "EMAIL", em))
        inn = gen_inn()
        test_texts.append((random.choice(MANUAL_TEMPLATES["INN"]).format(INN=inn), "INN", inn))
        cd = gen_card()
        test_texts.append((random.choice(MANUAL_TEMPLATES["CREDIT_CARD"]).format(CARD=cd), "CREDIT_CARD", cd))
        pp = gen_passport()
        test_texts.append((random.choice(MANUAL_TEMPLATES["PASSPORT"]).format(PASSPORT=pp), "PASSPORT", pp))
        og = gen_ogrnip()
        test_texts.append((random.choice(MANUAL_TEMPLATES["OGRNIP"]).format(OGRNIP=og), "OGRNIP", og))
        ip = gen_ip()
        test_texts.append((random.choice(MANUAL_TEMPLATES["IP"]).format(IP=ip), "IP", ip))
        bc = gen_birth_cert()
        test_texts.append((random.choice(MANUAL_TEMPLATES["BIRTH_CERT"]).format(BIRTH_CERT=bc), "BIRTH_CERT", bc))
        kl = gen_kladr()
        test_texts.append((random.choice(MANUAL_TEMPLATES["KLADR"]).format(KLADR=kl), "KLADR", kl))

    # --- Проверяем, что все сгенерированные значения валидны ---
    valid_data = True                                           # флаг: все ли тесты корректны
    for _, attr, value in test_texts:
        if attr == "INN" and not validate_inn(value):
            valid_data = False
        if attr == "CREDIT_CARD" and not validate_luhn(value):
            valid_data = False
        if attr == "PASSPORT" and not validate_passport_series(digits_only(value)[:4]):
            valid_data = False
        if attr == "OGRNIP" and not validate_ogrnip(value):
            valid_data = False
        if attr == "KLADR" and not validate_kladr(value):
            valid_data = False

    # --- Строим faker_samples с gold-спанами ---
    faker_samples = []
    for text, attr, value in test_texts:
        sp = _find_span(text, value)                            # позиция значения в тексте
        if sp:
            faker_samples.append((text, [(attr, *sp)]))         # (text, [(attr, start, end)])

    cnt_faker = {a: 0 for a in ATTRS}                           # счётчик текстов на атрибут
    for _, gold in faker_samples:
        for a, _, _ in gold:
            cnt_faker[a] += 1

    # --- Считаем метрики ---
    fact_samples = [(doc[0], doc[1]) for doc in factrueval_test_docs]  # factRuEval-тест
    per_attr_faker = compute_per_attr(predict, faker_samples)   # tp/fp/fn по Faker
    per_attr_fact = compute_per_attr(predict, fact_samples)     # tp/fp/fn по factRuEval
    per_attr_total = {                                          # суммарные tp/fp/fn
        a: {k: per_attr_faker[a][k] + per_attr_fact[a][k] for k in ("tp", "fp", "fn")}
        for a in ATTRS
    }

    rows = metrics_from_per_attr(per_attr_total)                # prec/rec/f1 по атрибутам

    # --- Weighted F1 и mean F1 ---
    weights = {a: per_attr_total[a]["tp"] + per_attr_total[a]["fn"] for a in ATTRS}  # вес = число истинных примеров
    tw = sum(weights.values())                                  # сумма весов
    if tw == 0:                                                 # нет ни одного gold-примера
        weighted_f1 = 0.0
    else:
        weighted_f1 = sum(rows[i][3] * weights[ATTRS[i]] for i in range(len(ATTRS))) / tw
    mean_f1 = sum(row[3] for row in rows) / len(ATTRS)          # простое среднее по атрибутам

    return {
        "rows":         rows,                                   # детальная таблица
        "weighted_f1":  weighted_f1,                            # взвешенное F1
        "mean_f1":      mean_f1,                                # среднее F1
        "valid_data":   valid_data,                             # все ли сгенерированные значения прошли контроль
        "cnt_faker":    cnt_faker,                              # по сколько текстов на атрибут
        "n_fact":       len(fact_samples),                      # сколько документов в factRuEval-тесте
        "n_faker":      len(faker_samples),                     # сколько текстов в Faker-тесте
    }