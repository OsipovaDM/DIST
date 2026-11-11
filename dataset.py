"""Загрузка и разметка factRuEval-2016"""
import builtins
import os
import random
import re
import urllib.request
import zipfile
from typing import List, Tuple

from config import FACT_RUEVAL_URL, DATA_DIR, TYPE_TO_ATTR, SEED

# ========== КОДИРОВКА ДЛЯ CORUS (Windows) ==========
def _utf8_open(path, *a, **kw):
    """open() с явной кодировкой UTF-8 — замена стандартному на Windows."""
    kw.setdefault("encoding", "utf-8")           # всегда UTF-8, если не передали другое
    return builtins.open(path, *a, **kw)


def _load_text_utf8(path):
    """Читает файл в UTF-8. Обёртка для monkey-patch corus.io.load_text."""
    with _utf8_open(path) as f:                  # открываем с UTF-8
        return f.read()


def _patch_corus_encoding():
    """Подменяет load_text в corus на UTF-8-версию.

    Нужно на Windows: corus открывает файлы без encoding, и Python подставляет
    cp1251 вместо UTF-8 → UnicodeDecodeError на кириллице. Патч идемпотентный:
    повторный вызов ничего не меняет.
    """
    import corus.io as _corus_io                  # модуль ввода-вывода corus
    import corus.sources.factru as _fr            # конкретный источник factRuEval
    _corus_io.load_text = _load_text_utf8         # глобальный load_text
    _fr.load_text = _load_text_utf8               # тот же load_text внутри factru


# ========== СКАЧИВАНИЕ И РАСПАКОВКА ==========
def download_factrueval(target_dir: str = DATA_DIR) -> str:
    """Скачивает и распаковывает factRuEval, если ещё не скачан.

    Возвращает путь к корневой папке корпуса. Идемпотентна: повторный вызов
    не качает заново, если папка уже существует.
    """
    os.makedirs(target_dir, exist_ok=True)                          # создаём папку data/
    marker = os.path.join(target_dir, "factRuEval-2016-master")     # маркер «уже распаковано»
    if os.path.exists(marker):                                      # архив уже распакован
        return marker
    zp = os.path.join(target_dir, "fr.zip")                         # путь к архиву
    if not os.path.exists(zp):                                      # архив не скачан
        urllib.request.urlretrieve(FACT_RUEVAL_URL, zp)             # качаем
    with zipfile.ZipFile(zp) as z:                                  # открываем архив
        z.extractall(target_dir)                                    # распаковываем в data/
    return marker


# ========== ПАРСИНГ ЗАПИСЕЙ CORUS ==========
def _rec_tokens(rec) -> List[Tuple[str, int, int]]:
    """Токены записи в формате (текст, начало, конец).

    Если corus дал токенизацию — используем её (это оригинальное деление
    из factRuEval). Если нет — режем по пробелам регуляркой.
    """
    toks = getattr(rec, "tokens", None)                             # токены из corus
    if toks:
        return [(t.text, t.start, t.stop) for t in toks]            # уже готовые триплеты
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", rec.text)]  # иначе — по пробелам


def rec_to_example(rec) -> Tuple[List[str], List[str]]:
    """Запись corus → (список токенов, список BIO-тегов).

    Для каждого токена проверяем, попадает ли он в span surname или name
    у объекта Person. Если да — ставим B- (первый токен сущности) или
    I- (продолжение). Продолжение определяется по предыдущему токену:
    если он был тем же атрибутом — I-, иначе B-.
    """
    toks = _rec_tokens(rec)                                         # токены с позициями
    tokens = [t[0] for t in toks]                                   # только текст токенов
    labels = ["O"] * len(tokens)                                    # по умолчанию всё — «не сущность»
    for i, (_, ts, te) in enumerate(toks):                          # проходим по токенам
        matched = None                                              # найденный атрибут
        for obj in rec.objects:                                     # перебираем объекты в записи
            if getattr(obj, "type", "") != "Person":                # только люди (ФИО)
                continue
            for span in obj.spans:                                  # перебираем спаны объекта
                if span.start <= ts < span.stop or span.start < te <= span.stop:  # токен пересекается со спаном
                    t = span.type.lower()                           # тип спана в нижнем регистре
                    if t in TYPE_TO_ATTR:                           # surname или name?
                        matched = TYPE_TO_ATTR[t]                   # маппим в SURNAME/NAME
                        break
            if matched:                                             # нашли — дальше не ищем
                break
        if matched:                                                 # токен принадлежит сущности
            prev = labels[i - 1] if i > 0 else "O"                  # метка предыдущего токена
            labels[i] = ("I-" if prev.endswith(matched) else "B-") + matched  # I- если та же сущность
    return tokens, labels


def rec_to_test_doc(rec) -> Tuple[str, List[Tuple[str, int, int]]]:
    """Запись corus → (полный текст, список gold-спанов).

    Для оценки нужны символьные позиции спанов в исходном тексте,
    а не BIO-теги. Возвращаем всё, что относится к ФИО.
    """
    gold = []                                                       # сюда собираем спаны
    for obj in rec.objects:                                         # перебираем объекты
        if getattr(obj, "type", "") != "Person":                    # только люди
            continue
        for span in obj.spans:                                      # перебираем спаны
            t = span.type.lower()                                   # тип в нижнем регистре
            if t in TYPE_TO_ATTR:                                   # surname или name?
                gold.append((TYPE_TO_ATTR[t], span.start, span.stop))  # сохраняем (ATTR, start, end)
    return rec.text, gold


# ========== ПУБЛИЧНЫЙ API ==========
def load_factrueval() -> Tuple[List[Tuple[List[str], List[str]]],
                               List[Tuple[str, List[Tuple[str, int, int]]]]]:
    """Загружает factRuEval и возвращает (examples, factrueval_test_docs).

    examples            — список (tokens, BIO-labels) для обучения/валидации (devset);
    factrueval_test_docs — список (text, gold_spans) для независимой оценки (testset).

    Скачать и распаковать архив (если ещё нет).
    Применить UTF-8 patch для corus (Windows).
    Прочитать devset и testset раздельно (два вызова load_factru).
    Преобразовать в примеры и gold-спаны.
    Если testset пуст/без Person — отрезать 20% devset в тест.
    Если devset без ФИО — подставить ручные примеры.
    """
    _patch_corus_encoding()                                         # фикс кодировки до первого чтения
    from corus import load_factru                                   # импорт после патча

    fact_dir = download_factrueval()                                # скачиваем/распаковываем
    dev_records = list(load_factru(fact_dir, sets=["devset"]))      # devset — для обучения
    test_records = list(load_factru(fact_dir, sets=["testset"]))    # testset — независимый тест

    examples = [rec_to_example(r) for r in dev_records if _rec_tokens(r)]  # BIO-примеры из devset
    factrueval_test_docs = [d for d in (rec_to_test_doc(r) for r in test_records) if d[1]]  # gold-спаны из testset

    if not factrueval_test_docs and dev_records:                    # testset пуст/без Person
        rng = random.Random(SEED)                                   # воспроизводимое перемешивание
        shuffled = list(dev_records)                                # копия devset
        rng.shuffle(shuffled)                                       # перемешиваем
        n = max(1, int(0.2 * len(shuffled)))                        # 20% в тест (минимум 1)
        dev_for_test = shuffled[:n]                                 # первые n — в тест
        dev_records = shuffled[n:]                                  # остальные — в обучение
        factrueval_test_docs = [d for d in (rec_to_test_doc(r) for r in dev_for_test) if d[1]]  # gold-спаны
        examples = [rec_to_example(r) for r in dev_records if _rec_tokens(r)]  # пересобираем train

    return examples, factrueval_test_docs                           # кортеж из двух списков