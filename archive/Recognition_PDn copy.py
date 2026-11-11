#%% [markdown]
# # Лабораторная работа: Распознавание ПДн в русскоязычных текстах
# Комбинированный подход: BERT (ФИО) + правила (6 структурированных атрибутов).
# Формат вывода: список кортежей `[("ATTR", start_idx, end_idx), ...]`.
!pip install --upgrade pip setuptools wheel
!pip install seqeval --use-pep517
!pip install transformers==4.46.3 datasets evaluate seqeval faker torch accelerate scikit-learn corus
#%%
# ============================================================
# Ячейка 1. Импорты, конфигурация, константы
# ============================================================
import os, re, sys, json, random, zipfile, urllib.request, shutil
from typing import List, Tuple, Dict, Optional, Set

# --- при первом запуске раскомментировать ---
# !pip install -q transformers datasets evaluate seqeval faker torch accelerate

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
)
from datasets import Dataset as HFDataset, DatasetDict, Features, Sequence, Value, ClassLabel
import evaluate
from faker import Faker

# ---- воспроизводимость ----
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---- глобальные константы ----
BASE_MODEL = "DeepPavlov/rubert-base-cased"
FACT_RUEVAL_URL = "https://github.com/dialogue-evaluation/factRuEval-2016/archive/master.zip"
DATA_DIR = "./data"
MODEL_OUT = "./rubert_pii"
MAX_LEN = 256
MAX_TRAIN_SAMPLES = 4000      # ограничение для скорости (None — без ограничения)
MAX_EVAL_SAMPLES  = 800
NUM_EPOCHS = 10
BATCH_SIZE = 16
LR = 2e-5
WEIGHT_DECAY = 0.01

# ---- атрибуты ПДн ----
ATTRS = ["SURNAME","NAME","PATRONYMIC","EMAIL","INN","PHONE","CREDIT_CARD","SNILS","PASSPORT"]
LABELS = ["O"] + [f"{p}-{a}" for a in ATTRS for p in ("B","I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# Коды операторов/регионов РФ для телефонов
OPERATOR_CODES: Set[str] = (
    {str(n) for n in range(900, 1000)}
    | {"495","499","812","383","343","351","861","863","831","846","847","844",
       "473","472","842","879","865","862","872","873","877","878"}
)

# Белый список регионов РФ для серии паспорта (первые 2 цифры)
PASSPORT_REGION_WHITELIST: Set[int] = {
    1,3,4,5,7,8,10,11,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,
    32,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
    54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,
    75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,
    96,97,98,99
}

print("Конфигурация загружена. Устройство:", "cuda" if torch.cuda.is_available() else "cpu")

#%%
# ============================================================
# Ячейка 2. Функции проверки контрольных разрядов
# ============================================================

def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)

def validate_inn(digits: str) -> bool:
    """Проверка 12-значного ИНН по алгоритму ФНС."""
    d = _digits_only(digits)
    if len(d) != 12:
        return False
    nums = [int(c) for c in d]
    c11 = [7,2,4,10,3,5,9,4,6,8]
    c12 = [3,7,2,4,10,3,5,9,4,6,8]
    n11 = (sum(nums[i]*c11[i] for i in range(10)) % 11) % 10
    if n11 != nums[10]:
        return False
    n12 = (sum(nums[i]*c12[i] for i in range(11)) % 11) % 10
    return n12 == nums[11]

def validate_snils(digits: str) -> bool:
    """Проверка контрольного числа СНИЛС (11 цифр)."""
    d = _digits_only(digits)
    if len(d) != 11:
        return False
    nums = [int(c) for c in d[:9]]
    ctrl = int(d[9:])
    s = sum(n*(9-i) for i, n in enumerate(nums))
    if s < 100:
        expected = s
    elif s in (100, 101):
        expected = 0
    else:
        expected = s % 101
        if expected == 100:
            expected = 0
    return expected == ctrl

def validate_luhn(card_number: str) -> bool:
    """Проверка номера карты по алгоритму Луна."""
    d = _digits_only(card_number)
    if len(d) != 16:
        return False
    total = 0
    for i, ch in enumerate(reversed(d)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0

def validate_passport_series(series: str) -> bool:
    """Проверка серии паспорта РФ по белому списку кодов регионов."""
    s = _digits_only(series)
    if len(s) != 4:
        return False
    return int(s[:2]) in PASSPORT_REGION_WHITELIST

# Быстрые проверки
assert validate_inn("500100732259") is True or validate_inn("770698959438") is True  # эталон из методички
assert validate_snils("11223344595") is True
assert validate_luhn("4561261212345467") is True
assert validate_passport_series("4506") is True
assert validate_passport_series("0000") is False
print("Контрольные функции OK.")

#%%
# ============================================================
# Ячейка 3. Загрузка и парсинг factRuEval-2016 → train/valid (devset) + test (testset) → пары (токен, BIO-тег)
# ============================================================
import os, re, zipfile, urllib.request, builtins

# --- фикс кодировки ДО импорта corus ---
def _utf8_open(path, *a, **kw):
    kw.setdefault("encoding", "utf-8")
    return builtins.open(path, *a, **kw)

def _load_text_utf8(path):
    with _utf8_open(path) as f:
        return f.read()

import corus.io as _corus_io
_corus_io.load_text = _load_text_utf8
import corus.sources.factru as _fr
_fr.load_text = _load_text_utf8

from corus import load_factru

def download_factrueval(target_dir="./data"):
    os.makedirs(target_dir, exist_ok=True)
    marker = os.path.join(target_dir, "factRuEval-2016-master")
    if os.path.exists(marker):
        return marker
    zp = os.path.join(target_dir, "fr.zip")
    if not os.path.exists(zp):
        urllib.request.urlretrieve(
            "https://github.com/dialogue-evaluation/factRuEval-2016/archive/master.zip", zp)
    with zipfile.ZipFile(zp) as z:
        z.extractall(target_dir)
    return marker

FACT_DIR = download_factrueval()
print("factRuEval:", FACT_DIR)

TYPE_TO_ATTR = {"surname": "SURNAME", "name": "NAME", "patronymic": "PATRONYMIC"}

dev_records, test_records = [], []
for rec in load_factru(FACT_DIR):
    url = (getattr(rec, "url", "") or "").replace("\\", "/")
    (test_records if "testset" in url else dev_records).append(rec)
print(f"devset: {len(dev_records)} док., testset: {len(test_records)} док.")

def _rec_tokens(rec):
    toks = getattr(rec, "tokens", None)
    if toks:
        return [(t.text, t.start, t.stop) for t in toks]
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", rec.text)]

def rec_to_example(rec):
    toks = _rec_tokens(rec)
    tokens = [t[0] for t in toks]
    labels = ["O"] * len(tokens)
    for i, (_, ts, te) in enumerate(toks):
        matched = None
        for obj in rec.objects:
            if getattr(obj, "type", "") != "Person":
                continue
            for span in obj.spans:
                if span.start <= ts < span.stop or span.start < te <= span.stop:
                    t = span.type.lower()
                    if t in TYPE_TO_ATTR:
                        matched = TYPE_TO_ATTR[t]; break
            if matched: break
        if matched:
            prev = labels[i-1] if i > 0 else "O"
            labels[i] = ("I-" if prev.endswith(matched) else "B-") + matched
    return tokens, labels

examples = [rec_to_example(r) for r in dev_records if _rec_tokens(r)]

def rec_to_test_doc(rec):
    gold = []
    for obj in rec.objects:
        if getattr(obj, "type", "") != "Person":
            continue
        for span in obj.spans:
            t = span.type.lower()
            if t in TYPE_TO_ATTR:
                gold.append((TYPE_TO_ATTR[t], span.start, span.stop))
    return rec.text, gold

factrueval_test_docs = [d for d in (rec_to_test_doc(r) for r in test_records) if d[1]]
print(f"test factRuEval: {len(factrueval_test_docs)} док. с gold-спанами")

if not any(l != "O" for _, labs in examples for l in labs):
    print("⚠ devset без ФИО — ручной fallback")
    MANUAL = [
        ("Петров Иван Сергеевич пришёл.", ["B-SURNAME","B-NAME","B-PATRONYMIC","O","O"]),
        ("Сидорова Мария Петровна ушла.", ["B-SURNAME","B-NAME","B-PATRONYMIC","O"]),
        ("Иванов Пётр написал письмо.",   ["B-SURNAME","B-NAME","O","O","O"]),
        ("Екимова Александра — юрист.",   ["B-SURNAME","B-NAME","O","O"]),
        ("Дмитрий Медведев выступил.",     ["B-NAME","B-SURNAME","O"]),
    ]
    examples = [(t.split(), l) for t, l in MANUAL]

print(f"Train/valid: {len(examples)} документов")

#%%
# ============================================================
# Ячейка 4. HF Dataset, токенизация WordPiece, метрики seqeval
# ============================================================

def to_hf(examples: List[Tuple[List[str], List[str]]]) -> HFDataset:
    data = {"tokens": [e[0] for e in examples],
            "ner_tags": [[LABEL2ID[l] for l in e[1]] for e in examples]}
    features = Features({
        "tokens":   Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=LABELS)),
    })
    return HFDataset.from_dict(data, features=features)

full_ds = to_hf(examples).shuffle(seed=SEED)
if MAX_TRAIN_SAMPLES:
    full_ds = full_ds.select(range(min(len(full_ds), MAX_TRAIN_SAMPLES)))
split = full_ds.train_test_split(test_size=0.2, seed=SEED)
raw = DatasetDict({"train": split["train"], "valid": split["test"]})
print(raw)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def tokenize_and_align(batch):
    enc = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True,
                    max_length=MAX_LEN)
    all_labels = batch["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = enc.word_ids(batch_index=i)
        prev = None
        seq = []
        for wid in word_ids:
            if wid is None:
                seq.append(-100)
            elif wid != prev:
                seq.append(labels[wid])
            else:
                lab = labels[wid]
                # внутренние сабтокены → I-XXX
                if LABELS[lab].startswith("B-"):
                    seq.append(LABEL2ID["I-" + LABELS[lab][2:]])
                else:
                    seq.append(lab)
            prev = wid
        new_labels.append(seq)
    enc["labels"] = new_labels
    return enc

tok_ds = raw.map(tokenize_and_align, batched=True,
                 remove_columns=raw["train"].column_names)
print("Токенизировано:", tok_ds)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

seqeval = evaluate.load("seqeval")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    true_pred = [[LABELS[p] for p, l in zip(pred, lab) if l != -100]
                 for pred, lab in zip(preds, labels)]
    true_lab  = [[LABELS[l] for p, l in zip(pred, lab) if l != -100]
                 for pred, lab in zip(preds, labels)]
    res = seqeval.compute(predictions=true_pred, references=true_lab,
                          zero_division=0, scheme="IOB2")
    return {
        "precision": res["overall_precision"],
        "recall":    res["overall_recall"],
        "f1":        res["overall_f1"],
        "accuracy":  res["overall_accuracy"],
    }

#%%
# ============================================================
# Ячейка 5. Обучение BERT (Trainer + TrainingArguments)
# ============================================================
model = AutoModelForTokenClassification.from_pretrained(
    BASE_MODEL, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
)

training_args = TrainingArguments(
    output_dir=MODEL_OUT,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to=None,
    save_safetensors=True,
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok_ds["train"],
    eval_dataset=tok_ds["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
print("Метрики на валидации:", trainer.evaluate())
trainer.save_model(MODEL_OUT)
tokenizer.save_pretrained(MODEL_OUT)

#%% [markdown]
# ## Правила для 6 структурированных атрибутов
# Работают и независимо от модели — приоритет при конфликте за правилами.

#%%
# ============================================================
# Ячейка 6. Правила (regex + контрольные разряды + контекст)
# ============================================================

# --- EMAIL ---
EMAIL_RE = re.compile(
    r"(?i)([a-z0-9а-яё_.+\-]+@[a-zа-яё\d][a-zа-яё\d.\-]*\.[a-zа-яё]{2,})"
)

# --- PHONE ---
# Префикс +7/8/7 (опц.), код оператора 3-4 цифры, 7 цифр; разделители: пробел, дефис, точка, скобки
PHONE_RE = re.compile(r"""
    (?<![\d])
    (?:
        (?:\+7|8|7)\s*
        (?:\(\s*)?
    )?
    (?:\(\s*)?(\d{3,4})(?:\s*\))?
    [\s\-\.]*
    (\d{3})[\s\-\.]*(\d{2})[\s\-\.]*(\d{2})
    (?![\d])
""", re.VERBOSE)

def extract_phones(text: str) -> List[Tuple[str,int,int]]:
    out = []
    for m in PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        # вариант с префиксом +7/8 → 11 цифр; без префикса → 10
        if len(digits) == 11:
            core = digits[1:]
        elif len(digits) == 10:
            core = digits
        else:
            continue
        op = core[:3] if core[:3] in OPERATOR_CODES else (core[:4] if core[:4] in OPERATOR_CODES else None)
        if op is None:
            continue
        # 7 цифр после оператора
        tail = core[len(op):]
        if len(tail) != 7:
            continue
        out.append(("PHONE", m.start(), m.end()))
    return out

# --- INN (12 цифр, разделители допустимы) ---
INN_RE = re.compile(r"(?<!\d)(\d[\s\-]?){11}\d(?!\d)")
def extract_inn(text: str) -> List[Tuple[str,int,int]]:
    out = []
    for m in INN_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) == 12 and validate_inn(digits):
            out.append(("INN", m.start(), m.end()))
    return out

# --- SNILS (XXX-XXX-XXX YY; разделители: пробел, дефис, точка) ---
SNILS_RE = re.compile(r"(?<!\d)(\d{3})[\s\-\.]?(\d{3})[\s\-\.]?(\d{3})[\s\-\.]?(\d{2})(?!\d)")
def extract_snils(text: str) -> List[Tuple[str,int,int]]:
    out = []
    for m in SNILS_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) == 11 and validate_snils(digits):
            out.append(("SNILS", m.start(), m.end()))
    return out

# --- CREDIT CARD (16 цифр, разделители: пробел, дефис) ---
CARD_RE = re.compile(r"(?<!\d)(?:\d[\s\-]?){15}\d(?!\d)")
def extract_card(text: str) -> List[Tuple[str,int,int]]:
    out = []
    for m in CARD_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) == 16 and validate_luhn(digits):
            out.append(("CREDIT_CARD", m.start(), m.end()))
    return out

# --- PASSPORT (серия 4 цифры + номер 6 цифр; разделители: пробел, дефис, без) ---
PASSPORT_RE = re.compile(r"(?<!\d)(\d{2})[\s\-]?(\d{2})[\s\-]?(\d{6})(?!\d)")
def extract_passport(text: str) -> List[Tuple[str,int,int]]:
    out = []
    for m in PASSPORT_RE.finditer(text):
        series = m.group(1) + m.group(2)
        if validate_passport_series(series):
            out.append(("PASSPORT", m.start(), m.end()))
    return out

def extract_by_rules(text: str) -> List[Tuple[str,int,int]]:
    """Все 6 структурированных атрибутов через регулярные выражения + контроль."""
    if not text:
        return []
    res = []
    res += [("EMAIL", m.start(), m.end()) for m in EMAIL_RE.finditer(text)]
    res += extract_phones(text)
    res += extract_inn(text)
    res += extract_snils(text)
    res += extract_card(text)
    res += extract_passport(text)
    return res

# --- смоук-тест правил ---
demo = ("Иван, тел +7 (999) 777-12-34, почта ivan@mail.ru, ИНН 770698959438, "
        "СНИЛС 112-233-445 95, карта 4561 2612 1234 5467, паспорт 45 06 123456")
print("Правила на демо:", extract_by_rules(demo))

#%%
# ============================================================
# Ячейка 7. Объединение ИИ и правил, функция predict(text)
# ============================================================
from transformers import pipeline as hf_pipeline

# Загружаем обученную модель для инференса (если она есть)
try:
    ner_pipe = hf_pipeline(
        "token-classification", model=MODEL_OUT, tokenizer=MODEL_OUT,
        aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1,
    )
    MODEL_READY = True
except Exception as e:
    print("Модель не загружена, работаем только по правилам:", e)
    ner_pipe = None
    MODEL_READY = False

def model_entities(text: str) -> List[Tuple[str,int,int]]:
    """Прогон текста через BERT, возврат (attr, start, end) для ФИО."""
    if not text or ner_pipe is None:
        return []
    out = []
    for ent in ner_pipe(text[:2000]):
        label = ent["entity_group"]  # NAME / SURNAME / PATRONYMIC
        if label in ("NAME","SURNAME","PATRONYMIC"):
            out.append((label, int(ent["start"]), int(ent["end"])))
    return out

def _overlap(a, b):
    return not (a[2] <= b[1] or a[1] >= b[2])

def merge_entities(rule_ents: List[Tuple[str,int,int]],
                   model_ents: List[Tuple[str,int,int]]) -> List[Tuple[str,int,int]]:
    """Правило > модель при конфликте; при одинаковом атрибуте — длиннее."""
    all_ents = [(l,s,e,"rule")  for l,s,e in rule_ents] + \
               [(l,s,e,"model") for l,s,e in model_ents]
    # сначала правила, затем более длинные
    all_ents.sort(key=lambda x: (0 if x[3]=="rule" else 1, -(x[2]-x[1])))
    result, used = [], []
    for l,s,e,src in all_ents:
        conflict = any(_overlap((l,s,e), (ul,us,ue)) for ul,us,ue in used)
        if conflict:
            # при одинаковом атрибуте — оставляем более длинный
            same = [(ul,us,ue) for ul,us,ue in used if ul == l and _overlap((l,s,e),(ul,us,ue))]
            if same:
                longer = max(same, key=lambda x: x[2]-x[1])
                if (e - s) > (longer[2] - longer[1]):
                    used = [(ul,us,ue) for ul,us,ue in used if (ul,us,ue) != longer]
                    result = [(ul,us,ue) for ul,us,ue in result if (ul,us,ue) != longer]
                    result.append((l,s,e)); used.append((l,s,e))
            continue
        result.append((l,s,e)); used.append((l,s,e))
    result.sort(key=lambda x: x[1])
    return result

def predict(text: str) -> List[Tuple[str,int,int]]:
    """Финальная функция: список (ATTR, start, end) в исходном тексте."""
    if not text or not text.strip():
        return []
    rule_ents  = extract_by_rules(text)
    model_ents = model_entities(text)
    return merge_entities(rule_ents, model_ents)

# Смоук-тест
print(predict("Иван Петров, тел +7 (999) 777-12-34, ivan@mail.ru, ИНН 770698959438."))
print(predict(""))
print(predict("Тут нет персональных данных вообще."))

#%%
# ============================================================
# Ячейка 8. Генерация тестового набора (Faker + шаблоны, ≥50 на атрибут)
# ============================================================
fake_ru = Faker("ru_RU")

# --- ВСТАВЬТЕ СЮДА ДОПОЛНИТЕЛЬНЫЕ РУЧНЫЕ ШАБЛОНЫ ПРИ НЕОБХОДИМОСТИ ---
MANUAL_TEMPLATES: Dict[str, List[str]] = {
    "SURNAME":    ["{SURNAME} пришёл на встречу.", "Мы позвонили {SURNAME_OBL}."],
    "NAME":       ["{NAME} написал письмо.",       "Передайте привет {NAME}."],
    "PATRONYMIC": ["{PATRONYMIC} подтвердил заказ.", "Обратитесь к {PATRONYMIC}."],
    "EMAIL":      ["Пишите на {EMAIL} до пятницы.",   "Резервный адрес: {EMAIL}."],
    "INN":        ["ИНН компании: {INN}.",            "Укажите ИНН {INN} в декларации."],
    "PHONE":      ["Звоните: {PHONE}.",               "Контакт: {PHONE}."],
    "CREDIT_CARD":["Оплата на карту {CARD}.",         "Реквизиты: {CARD}."],
    "SNILS":      ["СНИЛС: {SNILS}.",                 "Номер СНИЛС — {SNILS}."],
    "PASSPORT":   ["Паспорт: {PASSPORT}.",            "Серия и номер: {PASSPORT}."],
}

# --- вспомогательные генераторы валидных значений ---
def gen_inn() -> str:
    while True:
        base = [random.randint(0,9) for _ in range(10)]
        c11 = [7,2,4,10,3,5,9,4,6,8]
        n11 = (sum(base[i]*c11[i] for i in range(10)) % 11) % 10
        d11 = base + [n11]
        c12 = [3,7,2,4,10,3,5,9,4,6,8]
        n12 = (sum(d11[i]*c12[i] for i in range(11)) % 11) % 10
        res = "".join(map(str, base + [n11, n12]))
        if validate_inn(res):
            return res

def gen_snils() -> str:
    while True:
        nums = [random.randint(0,9) for _ in range(9)]
        s = sum(n*(9-i) for i, n in enumerate(nums))
        if s < 100: ctrl = s
        elif s in (100, 101): ctrl = 0
        else:
            ctrl = s % 101
            if ctrl == 100: ctrl = 0
        res = "".join(map(str, nums)) + f"{ctrl:02d}"
        if validate_snils(res):
            return res

def gen_card() -> str:
    while True:
        body = "4" + "".join(str(random.randint(0,9)) for _ in range(14))
        for cd in range(10):
            cand = body + str(cd)
            if validate_luhn(cand):
                return cand

def gen_phone() -> str:
    op = random.choice(list(OPERATOR_CODES))
    rest = "".join(str(random.randint(0,9)) for _ in range(7))
    num = op + rest  # 10 цифр
    fmt = random.choice(["dashed","spaced","parens","compact","dotted"])
    if fmt == "compact":
        return random.choice(["+7","8"]) + num
    if fmt == "spaced":
        return f"{random.choice(['+7','8'])} {op} {rest[:3]} {rest[3:5]} {rest[5:]}"
    if fmt == "dashed":
        return f"{random.choice(['+7','8'])}-{op}-{rest[:3]}-{rest[3:5]}-{rest[5:]}"
    if fmt == "parens":
        return f"+7 ({op}) {rest[:3]}-{rest[3:5]}-{rest[5:]}"
    return f"{random.choice(['+7','8'])}.{op}.{rest[:3]}.{rest[3:5]}.{rest[5:]}"

def gen_passport() -> str:
    region = random.choice(sorted(PASSPORT_REGION_WHITELIST))
    dept   = random.randint(0, 99)
    series = f"{region:02d}{dept:02d}"
    number = "".join(str(random.randint(0,9)) for _ in range(6))
    fmt = random.choice(["space","dash","compact"])
    if fmt == "space":   return f"{series[:2]} {series[2:]} {number}"
    if fmt == "dash":    return f"{series[:2]}-{series[2:]}-{number}"
    return series + number

def gen_email() -> str:
    return fake_ru.ascii_free_email()

def gen_name() -> str:      return fake_ru.first_name_male()
def gen_surname() -> str:   return fake_ru.last_name_male()
def gen_patronymic() -> str:return fake_ru.middle_name_male()

# Собираем тестовый корпус: ≥ 50 текстов на каждый атрибут
N_PER_ATTR = 60
test_texts: List[Tuple[str, str, str]] = []  # (text, attr, value) — value нужен для оценки спанов

def _add(text: str, attr: str, value: str):
    test_texts.append((text, attr, value))

for _ in range(N_PER_ATTR):
    # ФИО
    n, s, p = gen_name(), gen_surname(), gen_patronymic()
    tmpl = random.choice(MANUAL_TEMPLATES["SURNAME"])
    _add(tmpl.format(SURNAME=s, SURNAME_OBL=s + "у"), "SURNAME", s)
    tmpl = random.choice(MANUAL_TEMPLATES["NAME"])
    _add(tmpl.format(NAME=n), "NAME", n)
    tmpl = random.choice(MANUAL_TEMPLATES["PATRONYMIC"])
    _add(tmpl.format(PATRONYMIC=p), "PATRONYMIC", p)
    # EMAIL
    em = gen_email(); tmpl = random.choice(MANUAL_TEMPLATES["EMAIL"])
    _add(tmpl.format(EMAIL=em), "EMAIL", em)
    # INN
    inn = gen_inn(); tmpl = random.choice(MANUAL_TEMPLATES["INN"])
    _add(tmpl.format(INN=inn), "INN", inn)
    # PHONE
    ph = gen_phone(); tmpl = random.choice(MANUAL_TEMPLATES["PHONE"])
    _add(tmpl.format(PHONE=ph), "PHONE", ph)
    # CARD
    cd = gen_card(); tmpl = random.choice(MANUAL_TEMPLATES["CREDIT_CARD"])
    _add(tmpl.format(CARD=cd), "CREDIT_CARD", cd)
    # SNILS
    sn = gen_snils(); tmpl = random.choice(MANUAL_TEMPLATES["SNILS"])
    _add(tmpl.format(SNILS=sn), "SNILS", sn)
    # PASSPORT
    pp = gen_passport(); tmpl = random.choice(MANUAL_TEMPLATES["PASSPORT"])
    _add(tmpl.format(PASSPORT=pp), "PASSPORT", pp)

# Контроль количества
from collections import Counter
cnt = Counter(a for _, a, _ in test_texts)
print("Количество тестовых текстов по атрибутам:", dict(cnt))
assert all(cnt[a] >= 50 for a in ATTRS), "Не хватает тестов по атрибутам!"

#%%
# ============================================================
# Ячейка 9. Оценка качества, метрики (precision/recall/F1), assert'ы на ДВУХ тестах: factRuEval testset + Faker
# ============================================================
from collections import Counter
import re

def compute_per_attr(predict_fn, samples):
    """samples: list[(text, [(attr, start, end), ...])] → {attr: {tp, fp, fn}}"""
    per_attr = {a: {"tp":0, "fp":0, "fn":0} for a in ATTRS}
    for text, gold in samples:
        pset = set(predict_fn(text))
        gset = set(gold)
        for a, s, e in gset:
            if any(pa == a and not (e <= ps or s >= pe) for pa, ps, pe in pset):
                per_attr[a]["tp"] += 1
            else:
                per_attr[a]["fn"] += 1
        for a, s, e in pset:
            if not any(ga == a and not (e <= gs or s >= ge) for ga, gs, ge in gset):
                per_attr[a]["fp"] += 1
    return per_attr

def metrics_from_per_attr(per_attr):
    rows = []
    for a in ATTRS:
        tp, fp, fn = per_attr[a]["tp"], per_attr[a]["fp"], per_attr[a]["fn"]
        prec = tp/(tp+fp) if (tp+fp) else 0.0
        rec  = tp/(tp+fn) if (tp+fn) else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        rows.append((a, prec, rec, f1, tp, fp, fn))
    return rows

def weighted_f1(rows, weights):
    tw = sum(weights.values())
    if tw == 0: return 0.0, 0.0, 0.0
    wp = sum(rows[i][1]*weights[ATTRS[i]] for i in range(len(ATTRS))) / tw
    wr = sum(rows[i][2]*weights[ATTRS[i]] for i in range(len(ATTRS))) / tw
    wf = sum(rows[i][3]*weights[ATTRS[i]] for i in range(len(ATTRS))) / tw
    return wp, wr, wf

def _find_span(text, value):
    idx = text.find(value)
    return (idx, idx + len(value)) if idx >= 0 else None

# --- Набор 1: Faker (из ячейки 8) ---
faker_samples = []
for text, attr, value in test_texts:
    sp = _find_span(text, value)
    if sp:
        faker_samples.append((text, [(attr, *sp)]))

cnt_faker = Counter(attr for _, gold in faker_samples for attr, _, _ in gold)
print(f"Faker test: {len(faker_samples)} текстов")
print(f"  по атрибутам: {dict(cnt_faker)}")

# --- Набор 2: factRuEval testset ---
fact_samples = [(doc[3], doc[2]) for doc in factrueval_test_docs]
print(f"factRuEval test: {len(fact_samples)} документов")

# --- Оценка на каждом + агрегация ---
per_attr_faker = compute_per_attr(predict, faker_samples)
per_attr_fact  = compute_per_attr(predict, fact_samples)
per_attr_total = {a: {k: per_attr_faker[a][k] + per_attr_fact[a][k]
                      for k in ("tp","fp","fn")} for a in ATTRS}

rows = metrics_from_per_attr(per_attr_total)
print(f"\n{'ATTR':<14}{'prec':>8}{'rec':>8}{'f1':>8}{'tp':>6}{'fp':>6}{'fn':>6}")
for a, p, r, f, tp, fp, fn in rows:
    print(f"{a:<14}{p:>8.3f}{r:>8.3f}{f:>8.3f}{tp:>6}{fp:>6}{fn:>6}")

weights = {a: per_attr_total[a]["tp"] + per_attr_total[a]["fn"] for a in ATTRS}
wp, wr, wf = weighted_f1(rows, weights)
print(f"\nWeighted precision: {wp:.4f}")
print(f"Weighted recall   : {wr:.4f}")
print(f"Weighted F1       : {wf:.4f}")

# --- Разбивка по источникам ---
print("\n--- По источникам ---")
for name, pa, n in [("Faker",      per_attr_faker, len(faker_samples)),
                    ("factRuEval", per_attr_fact,  len(fact_samples))]:
    r = metrics_from_per_attr(pa)
    w = {a: pa[a]["tp"] + pa[a]["fn"] for a in ATTRS}
    _, _, f1d = weighted_f1(r, w)
    print(f"{name:<12} docs={n:<4} F1={f1d:.3f}")

# --- Проверка контрольных разрядов ---
def _all_checks():
    for _, attr, value in test_texts:
        if attr == "INN" and not validate_inn(value):
            raise AssertionError(f"Плохой ИНН: {value}")
        if attr == "SNILS" and not validate_snils(value):
            raise AssertionError(f"Плохой СНИЛС: {value}")
        if attr == "CREDIT_CARD" and not validate_luhn(value):
            raise AssertionError(f"Плохая карта: {value}")
        if attr == "PASSPORT":
            series = re.sub(r"\D", "", value)[:4]
            if not validate_passport_series(series):
                raise AssertionError(f"Плохая серия: {series}")
    return True
assert _all_checks()

# --- Приёмочные assert'ы ---
print("\n--- Приёмочные проверки ---")
assert len(fact_samples) > 0, "Не выполнен тест на factRuEval2016"
print(f"✅ Тест на factRuEval2016: {len(fact_samples)} документов")
assert all(cnt_faker[a] >= 50 for a in ATTRS), \
    f"Faker: <50 тестов на атрибут: {dict(cnt_faker)}"
print(f"✅ Faker: {len(faker_samples)} текстов, ≥50 на каждый атрибут")
assert wf >= 0.85, f"weighted F1 = {wf:.3f} < 0.85"
print(f"✅ Weighted F1 = {wf:.4f} ≥ 0.85")
print("✅ Контрольные разряды валидны")
print("\n🎉 Все приёмочные условия ЛР выполнены.")

#%%
# ============================================================
# ДИАГНОСТИКА (временная ячейка, не для сдачи)
# ============================================================
FIO_TEST = """Иванов Иван Иванович пришёл на работу раньше всех. Петрова Мария Сергеевна встретила его у входа. Сидоров Пётр Николаевич и Кузнецова Анна Дмитриевна обсуждали новый проект. Смирнов Алексей Владимирович подготовил отчёт. Морозова Елена Александровна разослала приглашения. Попов Дмитрий Андреевич и Соколова Ольга Игоревна провели переговоры. Лебедев Сергей Анатольевич выступил с докладом. Козлова Наталья Викторовна задала вопросы. Новиков Андрей Петрович и Егорова Татьяна Михайловна подвели итоги.

Вчера мы встретили Иванова Ивана Ивановича в кафе. Петровой Марии Сергеевне позвонили из офиса. Сидорову Петру Николаевичу поручили важное задание. Кузнецову Анну Дмитриевну отправили в командировку. Смирнову Алексею Владимировичу вручили награду. Морозовой Елене Александровне передали документы. Попова Дмитрия Андреевича пригласили на совещание. Соколовой Ольге Игоревне сообщили новость. Лебедева Сергея Анатольевича назначили руководителем. Козловой Наталье Викторовне выразили благодарность. Новикова Андрея Петровича и Егорову Татьяну Михайловну поздравили с праздником.

Руководитель спросил Иванова Ивана Ивановича о результатах. Мы гордимся Петровой Марией Сергеевной. Директор доволен Сидоровым Петром Николаевичем. Коллеги восхищаются Кузнецовой Анной Дмитриевной. Отчёт подготовлен Смирновым Алексеем Владимировичем. Письмо написано Морозовой Еленой Александровной. Проект представлен Поповым Дмитрием Андреевичем. Презентация сделана Соколовой Ольгой Игоревной. Доклад прочитан Лебедевым Сергеем Анатольевичем. Вопрос задан Козловой Натальей Викторовной. Итоги подведены Новиковым Андреем Петровичем и Егоровой Татьяной Михайловной.

Иванов, Петров, Сидоров, Кузнецов, Смирнов, Морозов, Попов, Соколов, Лебедев, Козлов, Новиков, Егоров, Васильев, Михайлов, Фёдоров, Николаев, Павлов, Семёнов. Ивановой, Петровой, Сидоровой, Кузнецовой, Смирновой, Морозовой, Поповой, Соколовой. Иванову, Петрову, Сидорову, Кузнецову, Смирнову, Морозову, Попову, Соколову. Ивановым, Петровым, Сидоровым, Кузнецовым, Смирновым, Морозовым. Иванове, Петрове, Сидорове, Кузнецове.

Иван, Пётр, Сергей, Алексей, Дмитрий, Андрей, Николай, Михаил, Владимир, Александр, Елена, Мария, Анна, Ольга, Наталья, Татьяна, Екатерина, Ирина, Светлана. Ивана, Петра, Сергея, Алексея, Дмитрия, Андрея, Николая, Михаила. Ивану, Петру, Сергею, Алексею, Дмитрию, Андрею, Николаю, Михаилу. Иваном, Петром, Сергеем, Алексеем, Дмитрием, Андреем, Николаем, Михаилом. Иване, Петре, Сергее, Алексее.

Иванович, Петрович, Сергеевич, Алексеевич, Дмитриевич, Андреевич, Николаевич, Михайлович. Ивановна, Петровна, Сергеевна, Алексеевна, Дмитриевна, Андреевна, Николаевна, Михайловна. Ивановича, Петровича, Сергеевича, Алексеевича. Ивановне, Петровне, Сергеевне, Алексеевне. Ивановной, Петровной, Сергеевной, Алексеевной. Ивановичем, Петровичем, Сергеевичем, Алексеевичем."""

print(f"Длина текста: {len(FIO_TEST)} символов, ~{len(FIO_TEST.split())} слов")

# ---- 1. Загружена ли модель? ----
print(f"\n[1] MODEL_READY = {MODEL_READY}, ner_pipe = {ner_pipe is not None}")
if not MODEL_READY:
    print("  ❌ МОДЕЛЬ НЕ ЗАГРУЖЕНА — вот причина F1=0 по ФИО!")
    print("  Проверьте, что папка ./rubert_pii существует и содержит config.json:")
    import os
    print(f"  {os.path.abspath(MODEL_OUT)} exists = {os.path.exists(MODEL_OUT)}")
    if os.path.exists(MODEL_OUT):
        print(f"  files: {os.listdir(MODEL_OUT)}")

# ---- 2. Что видит BERT напрямую? ----
if MODEL_READY:
    print(f"\n[2] Сырые предсказания BERT (первые 20):")
    raw = ner_pipe(FIO_TEST[:2000])
    from collections import Counter
    c = Counter(e["entity_group"] for e in raw)
    print(f"  Всего предсказаний: {len(raw)}")
    print(f"  По меткам: {dict(c)}")
    print(f"  Первые 20:")
    for e in raw[:20]:
        word = FIO_TEST[e["start"]:e["end"]]
        print(f"    {e['entity_group']:15s} '{word}'  score={e['score']:.3f}")

# ---- 3. Что делает model_entities? ----
print(f"\n[3] model_entities(FIO_TEST):")
mes = model_entities(FIO_TEST)
print(f"  Найдено: {len(mes)}")
for m in mes[:10]:
    print(f"    {m[0]:15s} '{FIO_TEST[m[1]:m[2]]}'")

# ---- 4. Что делает predict()? ----
print(f"\n[4] predict(FIO_TEST):")
pes = predict(FIO_TEST)
print(f"  Найдено: {len(pes)}")
from collections import Counter
print(f"  По атрибутам: {dict(Counter(p[0] for p in pes))}")
for p in pes[:20]:
    print(f"    {p[0]:15s} '{FIO_TEST[p[1]:p[2]]}' [{p[1]}:{p[2]}]")

# ---- 5. Метки в загруженной модели ----
if MODEL_READY:
    print(f"\n[5] id2label загруженной модели (первые 10):")
    for i in range(10):
        print(f"    {i}: {ner_pipe.model.config.id2label.get(i, '?')}")
# %%
#%%
# ============================================================
# ДИАГНОСТИКА 2: прямой прогон модели без pipeline
# ============================================================
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

tok = AutoTokenizer.from_pretrained(MODEL_OUT)
mdl = AutoModelForTokenClassification.from_pretrained(MODEL_OUT)
mdl.eval()

texts = [
    "Иванов Иван Иванович пришёл на работу.",
    "Петрова Мария Сергеевна встретила его у входа.",
    "Сидоров Пётр Николаевич и Кузнецова Анна Дмитриевна обсуждали проект.",
    "Иван написал письмо.",
    "Петрова ушла.",
    "Иванович подтвердил.",
]

for text in texts:
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = mdl(**inputs).logits
    preds = logits.argmax(-1).squeeze().tolist()
    tokens = tok.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    print(f"\nТекст: {text}")
    for t, p in zip(tokens, preds):
        if t in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        print(f"  {t:20s} → {mdl.config.id2label[p]}")

# ---- Проверка весов classifier-head ----
print("\n[ВЕСА] classifier.weight — статистика:")
w = mdl.classifier.weight.data
print(f"  shape: {w.shape}")
print(f"  mean: {w.mean().item():.6f}, std: {w.std().item():.6f}")
print(f"  min: {w.min().item():.6f}, max: {w.max().item():.6f}")
print(f"  Число строк с ненулевым std (признак обучения): {sum(1 for r in w if r.std() > 1e-4)}")

print("\n[ФАЙЛЫ] Что в папке модели:")
import os
for f in os.listdir(MODEL_OUT):
    p = os.path.join(MODEL_OUT, f)
    size = os.path.getsize(p) if os.path.isfile(p) else 0
    mtime = os.path.getmtime(p) if os.path.isfile(p) else 0
    from datetime import datetime
    print(f"  {f:30s} {size/1e6:8.2f} МБ  {datetime.fromtimestamp(mtime)}")
# %%
