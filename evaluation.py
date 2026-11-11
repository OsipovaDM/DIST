"""Оценка качества распознавания ПДн"""
import random
import re
from typing import List, Tuple

from faker import Faker

from config import (
    ATTRS, N_PER_ATTR, PASSPORT_REGION_WHITELIST,
    digits_only, MANUAL_TEMPLATES,
)
from recognize import (
    predict,
    validate_inn, validate_luhn, validate_passport_series,
    validate_ogrnip, validate_kladr,
)

fake_ru = Faker("ru_RU")   # генератор русских имён/фамилий/email

# ========== ШАБЛОНЫ И ГЕНЕРАТОРЫ ==========
def gen_inn(fmt: str = "compact") -> str:
    """Случайный валидный ИНН. fmt: compact | space | dash."""
    while True:
        base = [random.randint(0, 9) for _ in range(10)]
        c11 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        n11 = (sum(base[i] * c11[i] for i in range(10)) % 11) % 10
        d11 = base + [n11]
        c12 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        n12 = (sum(d11[i] * c12[i] for i in range(11)) % 11) % 10
        digits = "".join(map(str, base + [n11, n12]))
        if validate_inn(digits):
            break
    if fmt == "space":
        return f"{digits[:4]} {digits[4:7]} {digits[7:10]} {digits[10:]}"
    if fmt == "dash":
        return f"{digits[:4]}-{digits[4:7]}-{digits[7:10]}-{digits[10:]}"
    return digits


def gen_card(fmt: str = "compact") -> str:
    """Случайный валидный номер карты. fmt: compact | space | dash."""
    while True:
        body = "4" + "".join(str(random.randint(0, 9)) for _ in range(14))
        for cd in range(10):
            cand = body + str(cd)
            if validate_luhn(cand):
                digits = cand
                break
        else:
            continue
        break
    if fmt == "space":
        return f"{digits[:4]} {digits[4:8]} {digits[8:12]} {digits[12:]}"
    if fmt == "dash":
        return f"{digits[:4]}-{digits[4:8]}-{digits[8:12]}-{digits[12:]}"
    return digits


def gen_passport(fmt: str = "space") -> str:
    """Случайный паспорт. fmt: compact | space | dash."""
    region = random.choice(sorted(PASSPORT_REGION_WHITELIST))
    dept = random.randint(0, 99)
    series = f"{region:02d}{dept:02d}"
    number = "".join(str(random.randint(0, 9)) for _ in range(6))
    if fmt == "compact":
        return series + number
    if fmt == "dash":
        return f"{series[:2]}-{series[2:]}-{number}"
    return f"{series[:2]} {series[2:]} {number}"


def gen_ogrnip(fmt: str = "compact") -> str:
    """Случайный валидный ОГРНИП. fmt: compact | space | dash."""
    while True:
        body = "3" + "".join(str(random.randint(0, 9)) for _ in range(13))
        ctrl = (int(body) % 13) % 10
        digits = body + str(ctrl)
        if validate_ogrnip(digits):
            break
    if fmt == "space":
        return f"{digits[:3]} {digits[3:6]} {digits[6:9]} {digits[9:12]} {digits[12:]}"
    if fmt == "dash":
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:9]}-{digits[9:12]}-{digits[12:]}"
    return digits

def gen_ip() -> str:
    """Случайный IPv4: 4 октета 1–254 (без 0 и 255 — служебные)."""
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def gen_birth_cert(fmt: str = "plain") -> str:
    """Свидетельство о рождении. fmt: plain | subj_space | subj_dash."""
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
             "XI", "XII", "XIII", "XIV", "XV"]
    series = random.choice(roman)
    num = "".join(str(random.randint(0, 9)) for _ in range(6))
    subj = random.choice(["АК", "МЮ", "МК", "АБ", "БА", "ЮК"])
    if fmt == "subj_space":
        return f"{series}-{subj} {num}"
    if fmt == "subj_dash":
        return f"{series}-{subj}-{num}"
    return f"{series} {num}"


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
def _build_test_texts(n_per_attr: int) -> list:
    """Строит тестовые тексты с покрытием ВСЕХ шаблонов и ВСЕХ форматов.

    Для каждого атрибута генерируется n_per_attr текстов. Шаблоны
    и форматы разделителей перебираются по кругу, чтобы каждый вариант
    встретился примерно одинаковое число раз.
    """
    test_texts = []

    # SURNAME NAME
    for i in range(n_per_attr):
        n = gen_name()                                              # имя
        s = gen_surname()                                           # фамилия
        tmpl = MANUAL_TEMPLATES["FIO"][i % len(MANUAL_TEMPLATES["FIO"])]
        text = tmpl.format(NAME=n, SURNAME=s)                       # полное ФИО в контексте
        test_texts.append((text, "SURNAME", s))                     # проверяем фамилию
        test_texts.append((text, "NAME", n))                        # и имя в том же тексте

    # EMAIL — один формат, все шаблоны
    for i in range(n_per_attr):
        tmpl = MANUAL_TEMPLATES["EMAIL"][i % len(MANUAL_TEMPLATES["EMAIL"])]
        em = gen_email()
        test_texts.append((tmpl.format(EMAIL=em), "EMAIL", em))

    # INN — 3 формата × все шаблоны
    for i in range(n_per_attr):
        fmt = ["compact", "space", "dash"][i % 3]
        tmpl = MANUAL_TEMPLATES["INN"][(i // 3) % len(MANUAL_TEMPLATES["INN"])]
        inn = gen_inn(fmt=fmt)
        test_texts.append((tmpl.format(INN=inn), "INN", inn))

    # CREDIT_CARD — 3 формата × все шаблоны
    for i in range(n_per_attr):
        fmt = ["compact", "space", "dash"][i % 3]
        tmpl = MANUAL_TEMPLATES["CREDIT_CARD"][(i // 3) % len(MANUAL_TEMPLATES["CREDIT_CARD"])]
        cd = gen_card(fmt=fmt)
        test_texts.append((tmpl.format(CARD=cd), "CREDIT_CARD", cd))

    # PASSPORT — 3 формата × все шаблоны
    for i in range(n_per_attr):
        fmt = ["space", "dash", "compact"][i % 3]
        tmpl = MANUAL_TEMPLATES["PASSPORT"][(i // 3) % len(MANUAL_TEMPLATES["PASSPORT"])]
        pp = gen_passport(fmt=fmt)
        test_texts.append((tmpl.format(PASSPORT=pp), "PASSPORT", pp))

    # OGRNIP — 3 формата × все шаблоны
    for i in range(n_per_attr):
        fmt = ["compact", "space", "dash"][i % 3]
        tmpl = MANUAL_TEMPLATES["OGRNIP"][(i // 3) % len(MANUAL_TEMPLATES["OGRNIP"])]
        og = gen_ogrnip(fmt=fmt)
        test_texts.append((tmpl.format(OGRNIP=og), "OGRNIP", og))

    # IP — один формат, все шаблоны
    for i in range(n_per_attr):
        tmpl = MANUAL_TEMPLATES["IP"][i % len(MANUAL_TEMPLATES["IP"])]
        ip = gen_ip()
        test_texts.append((tmpl.format(IP=ip), "IP", ip))

    # BIRTH_CERT — 3 формата × все шаблоны
    for i in range(n_per_attr):
        fmt = ["plain", "subj_space", "subj_dash"][i % 3]
        tmpl = MANUAL_TEMPLATES["BIRTH_CERT"][(i // 3) % len(MANUAL_TEMPLATES["BIRTH_CERT"])]
        bc = gen_birth_cert(fmt=fmt)
        test_texts.append((tmpl.format(BIRTH_CERT=bc), "BIRTH_CERT", bc))

    # KLADR — один формат, все шаблоны
    for i in range(n_per_attr):
        tmpl = MANUAL_TEMPLATES["KLADR"][i % len(MANUAL_TEMPLATES["KLADR"])]
        kl = gen_kladr()
        test_texts.append((tmpl.format(KLADR=kl), "KLADR", kl))

    return test_texts


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
    test_texts = _build_test_texts(n_per_attr)

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