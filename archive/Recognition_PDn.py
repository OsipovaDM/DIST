#%% [markdown]
# # КМ2 (чётный вариант): Распознавание ПДн в русскоязычных текстах
# Комбинированный подход: BERT (ФИО: Фамилия, Имя) + правила (структурированные атрибуты).
# Атрибуты: SURNAME, NAME, INN, PASSPORT, CREDIT_CARD, OGRNIP, IP, EMAIL, BIRTH_CERT, KLADR

#%%
# ============================================================
# Ячейка 0. Установка зависимостей (один раз, потом Restart Kernel)
# ============================================================
import sys, subprocess
def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])

_pip("--upgrade", "pip", "setuptools", "wheel")
_pip("transformers==4.46.3", "datasets>=2.18,<3", "evaluate>=0.4,<1",
     "faker", "accelerate", "scikit-learn")
_pip("seqeval")
_pip("corus")
print("✅ Установка завершена. Перезапустите ядро (Kernel → Restart).")

#%%
# ============================================================
# Ячейка 1. Импорты, конфигурация, константы (чётный вариант КМ2)
# ============================================================
import os, re, sys, json, random, zipfile, urllib.request, shutil, builtins
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
import torch
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

# ---- константы ----
BASE_MODEL = "DeepPavlov/rubert-base-cased"
FACT_RUEVAL_URL = "https://github.com/dialogue-evaluation/factRuEval-2016/archive/master.zip"
DATA_DIR = "./data"
MODEL_OUT = "./rubert_pii"

MAX_LEN = 256
NUM_EPOCHS = 7
BATCH_SIZE = 16
LR = 3e-5
WEIGHT_DECAY = 0.01

# ---- атрибуты чётного варианта (без PATRONYMIC) ----
ATTRS = [
    "SURNAME", "NAME",
    "INN", "PASSPORT", "CREDIT_CARD",
    "OGRNIP", "IP", "EMAIL", "BIRTH_CERT", "KLADR",
]
LABELS = ["O"] + [f"{p}-{a}" for a in ATTRS for p in ("B","I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

PASSPORT_REGION_WHITELIST: Set[int] = {
    1,3,4,5,7,8,10,11,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,
    32,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
    54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,
    75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,
    96,97,98,99,
}

print("Конфигурация загружена. Устройство:", "cuda" if torch.cuda.is_available() else "cpu")
print(f"Атрибутов: {len(ATTRS)} | BIO-меток: {len(LABELS)}")

#%%
# ============================================================
# Ячейка 2. Функции проверки контрольных разрядов
# ============================================================
def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)

def validate_inn(digits: str) -> bool:
    d = _digits_only(digits)
    if len(d) != 12: return False
    nums = [int(c) for c in d]
    c11 = [7,2,4,10,3,5,9,4,6,8]
    c12 = [3,7,2,4,10,3,5,9,4,6,8]
    n11 = (sum(nums[i]*c11[i] for i in range(10)) % 11) % 10
    if n11 != nums[10]: return False
    n12 = (sum(nums[i]*c12[i] for i in range(11)) % 11) % 10
    return n12 == nums[11]

def validate_luhn(card_number: str) -> bool:
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
    s = _digits_only(series)
    if len(s) != 4: return False
    return int(s[:2]) in PASSPORT_REGION_WHITELIST

def validate_ogrnip(digits: str) -> bool:
    d = _digits_only(digits)
    if len(d) != 15: return False
    return (int(d[:14]) % 13) % 10 == int(d[14])

def validate_kladr(digits: str) -> bool:
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
print("Контрольные функции OK.")

#%%
# ============================================================
# Ячейка 3. factRuEval → train/valid (devset) + test (testset)
# Только SURNAME и NAME из corus, PATRONYMIC игнорируется
# ============================================================
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
        urllib.request.urlretrieve(FACT_RUEVAL_URL, zp)
    with zipfile.ZipFile(zp) as z:
        z.extractall(target_dir)
    return marker

FACT_DIR = download_factrueval()
print("factRuEval:", FACT_DIR)

# Только SURNAME и NAME — PATRONYMIC отбрасываем
TYPE_TO_ATTR = {"surname": "SURNAME", "name": "NAME"}

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
print(f"test factRuEval (из testset): {len(factrueval_test_docs)} док.")

# Fallback: если testset пуст/без Person — 20% devset в тест
if not factrueval_test_docs and dev_records:
    print("⚠ testset пуст → берём 20% devset как factRuEval-тест")
    rng = random.Random(SEED)
    shuffled = list(dev_records); rng.shuffle(shuffled)
    n = max(1, int(0.2 * len(shuffled)))
    dev_for_test = shuffled[:n]
    dev_records  = shuffled[n:]
    factrueval_test_docs = [d for d in (rec_to_test_doc(r) for r in dev_for_test) if d[1]]
    examples = [rec_to_example(r) for r in dev_records if _rec_tokens(r)]
    print(f"  Пересобрано: train/valid={len(examples)}, test={len(factrueval_test_docs)}")

if not any(l != "O" for _, labs in examples for l in labs):
    print("⚠ devset без ФИО — ручной fallback")
    MANUAL = [
        ("Петров Иван пришёл.",        ["B-SURNAME","B-NAME","O"]),
        ("Сидорова Мария ушла.",       ["B-SURNAME","B-NAME","O"]),
        ("Иванов Пётр написал письмо.", ["B-SURNAME","B-NAME","O","O"]),
        ("Екимова Александра — юрист.", ["B-SURNAME","B-NAME","O","O"]),
        ("Дмитрий Медведев выступил.", ["B-NAME","B-SURNAME","O"]),
    ]
    examples = [(t.split(), l) for t, l in MANUAL]

print(f"Train/valid: {len(examples)} документов")

#%%
# ============================================================
# Ячейка 4. HF Dataset, токенизация WordPiece, метрики seqeval
# ============================================================
def to_hf(examples):
    data = {"tokens": [e[0] for e in examples],
            "ner_tags": [[LABEL2ID[l] for l in e[1]] for e in examples]}
    features = Features({
        "tokens":   Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=LABELS)),
    })
    return HFDataset.from_dict(data, features=features)

full_ds = to_hf(examples).shuffle(seed=SEED)
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
# Ячейка 5. Обучение BERT (версионно-независимо)
# ============================================================
import inspect

RETRAIN = False
if RETRAIN and os.path.exists(MODEL_OUT):
    shutil.rmtree(MODEL_OUT, ignore_errors=True)
    print(f"Удалена старая модель: {MODEL_OUT}")

args_sig = set(inspect.signature(TrainingArguments.__init__).parameters)
eval_key = "eval_strategy" if "eval_strategy" in args_sig else "evaluation_strategy"

args_kwargs = {
    "output_dir": MODEL_OUT,
    eval_key: "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 1,
    "learning_rate": LR,
    "per_device_train_batch_size": BATCH_SIZE,
    "per_device_eval_batch_size": BATCH_SIZE,
    "num_train_epochs": NUM_EPOCHS,
    "weight_decay": WEIGHT_DECAY,
    "logging_steps": 50,
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "greater_is_better": True,
    "report_to": "none",
    "seed": SEED,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "dataloader_num_workers": 0,
}
if "save_safetensors" in args_sig:
    args_kwargs["save_safetensors"] = True

training_args = TrainingArguments(**args_kwargs)
print(f"transformers: {__import__('transformers').__version__}, используется '{eval_key}'")

model = AutoModelForTokenClassification.from_pretrained(
    BASE_MODEL, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)

trainer_sig = set(inspect.signature(Trainer.__init__).parameters)
tok_key = "processing_class" if "processing_class" in trainer_sig else "tokenizer"

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok_ds["train"],
    eval_dataset=tok_ds["valid"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    **{tok_key: tokenizer},
)

print("🚀 Старт обучения...")
trainer.train()
print("Метрики на валидации:", trainer.evaluate())
trainer.save_model(MODEL_OUT)
tokenizer.save_pretrained(MODEL_OUT)

total_bytes = sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, _, fs in os.walk(MODEL_OUT) for f in fs
)
print(f"💾 Сохранено в: {os.path.abspath(MODEL_OUT)} ({total_bytes/1e6:.1f} МБ)")

#%%
# ============================================================
# Ячейка 6. Правила для структурированных атрибутов
# ============================================================

EMAIL_RE = re.compile(
    r"(?i)([a-z0-9а-яё_.+\-]+@[a-zа-яё\d][a-zа-яё\d.\-]*\.[a-zа-яё]{2,})"
)

IP_RE = re.compile(
    r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)(?![\d.])"
)
def extract_ip(text):
    return [("IP", m.start(), m.end()) for m in IP_RE.finditer(text)]

INN_RE = re.compile(r"(?<!\d)(\d[\s\-]?){11}\d(?!\d)")
def extract_inn(text):
    out = []
    for m in INN_RE.finditer(text):
        if validate_inn(re.sub(r"\D", "", m.group(0))):
            out.append(("INN", m.start(), m.end()))
    return out

CARD_RE = re.compile(r"(?<!\d)(?:\d[\s\-]?){15}\d(?!\d)")
def extract_card(text):
    out = []
    for m in CARD_RE.finditer(text):
        if validate_luhn(re.sub(r"\D", "", m.group(0))):
            out.append(("CREDIT_CARD", m.start(), m.end()))
    return out

PASSPORT_RE = re.compile(r"(?<!\d)(\d{2})[\s\-]?(\d{2})[\s\-]?(\d{6})(?!\d)")
def extract_passport(text):
    out = []
    for m in PASSPORT_RE.finditer(text):
        if validate_passport_series(m.group(1) + m.group(2)):
            out.append(("PASSPORT", m.start(), m.end()))
    return out

OGRNIP_RE = re.compile(r"(?<!\d)(\d[\s\-]?){14}\d(?!\d)")
def extract_ogrnip(text):
    out = []
    for m in OGRNIP_RE.finditer(text):
        if validate_ogrnip(re.sub(r"\D", "", m.group(0))):
            out.append(("OGRNIP", m.start(), m.end()))
    return out

KLADR_RE = re.compile(r"(?<!\d)(\d{13}|\d{17})(?!\d)")
def extract_kladr(text):
    out = []
    for m in KLADR_RE.finditer(text):
        if validate_kladr(m.group(0)):
            out.append(("KLADR", m.start(), m.end()))
    return out

BIRTH_CERT_RE = re.compile(
    r"(?<![А-ЯЁA-Z\d])([IVX]{1,4})[\s\-]?(?:[А-ЯЁA-Z]{2}[\s\-]?)?(\d{6,7})(?!\d)"
)
def extract_birth_cert(text):
    return [("BIRTH_CERT", m.start(), m.end()) for m in BIRTH_CERT_RE.finditer(text)]

def extract_by_rules(text):
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

demo = ("Иван, почта ivan@mail.ru, ИНН 770698959438, "
        "карта 4561 2612 1234 5467, паспорт 45 06 123456, "
        "ОГРНИП 304500116000157, IP 192.168.1.100, "
        "КЛАДР 7700000000000, свидетельство IV-МЮ 123456")
print("Правила на демо:")
for x in extract_by_rules(demo):
    print(f"  {x[0]:15s} '{demo[x[1]:x[2]]}'")

#%%
# ============================================================
# Ячейка 7. predict(text) — объединение BERT + правил
# ============================================================
from transformers import pipeline as hf_pipeline

try:
    ner_pipe = hf_pipeline(
        "token-classification", model=MODEL_OUT, tokenizer=MODEL_OUT,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1,
    )
    
    MODEL_READY = True
except Exception as e:
    print("Модель не загружена:", e)
    ner_pipe = None
    MODEL_READY = False

def model_entities(text):
    if not text or ner_pipe is None:
        return []
    # Обрезаем текст по токенам до MAX_LEN, сохраняя границу в символах
    enc = tokenizer(
        text, truncation=True, max_length=MAX_LEN,
        add_special_tokens=False, return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"]
    if offsets:
        text = text[: offsets[-1][1]]
    out = []
    for ent in ner_pipe(text):
        label = ent["entity_group"]
        if label in ("NAME","SURNAME"):
            out.append((label, int(ent["start"]), int(ent["end"])))
    return out

def _overlap(a, b):
    return not (a[2] <= b[1] or a[1] >= b[2])

def merge_entities(rule_ents, model_ents):
    all_ents = [(l,s,e,"rule")  for l,s,e in rule_ents] + \
               [(l,s,e,"model") for l,s,e in model_ents]
    all_ents.sort(key=lambda x: (0 if x[3]=="rule" else 1, -(x[2]-x[1])))
    result, used = [], []
    for l,s,e,src in all_ents:
        conflict = any(_overlap((l,s,e), (ul,us,ue)) for ul,us,ue in used)
        if conflict:
            same = [(ul,us,ue) for ul,us,ue in used if ul == l and _overlap((l,s,e),(ul,us,ue))]
            if same:
                longer = max(same, key=lambda x: x[2]-x[1])
                if (e - s) > (longer[2] - longer[1]):
                    used = [u for u in used if u != longer]
                    result = [r for r in result if r != longer]
                    result.append((l,s,e)); used.append((l,s,e))
            continue
        result.append((l,s,e)); used.append((l,s,e))
    result.sort(key=lambda x: x[1])
    return result

def predict(text):
    if not text or not text.strip():
        return []
    return merge_entities(extract_by_rules(text), model_entities(text))

print("predict(''):", predict(""))
print("predict('Тут нет ПДн.'):", predict("Тут нет ПДн."))
print("predict('Иванов Иван, ivan@mail.ru, ИНН 770698959438.'):",
      predict("Иванов Иван, ivan@mail.ru, ИНН 770698959438."))

#%%
# ============================================================
# Ячейка 8. Faker-тест (≥50 на атрибут) + валидные генераторы
# ============================================================
fake_ru = Faker("ru_RU")

MANUAL_TEMPLATES = {
    "SURNAME":    ["{SURNAME} пришёл на встречу.", "Мы позвонили {SURNAME_OBL}."],
    "NAME":       ["{NAME} написал письмо.",       "Передайте привет {NAME}."],
    "EMAIL":      ["Пишите на {EMAIL} до пятницы.",   "Резервный адрес: {EMAIL}."],
    "INN":        ["ИНН компании: {INN}.",            "Укажите ИНН {INN} в декларации."],
    "CREDIT_CARD":["Оплата на карту {CARD}.",         "Реквизиты: {CARD}."],
    "PASSPORT":   ["Паспорт: {PASSPORT}.",            "Серия и номер: {PASSPORT}."],
    "OGRNIP":     ["ОГРНИП: {OGRNIP}.",               "Регистрация ИП: {OGRNIP}."],
    "IP":         ["IP-адрес сервера: {IP}.",         "Соединение с {IP}."],
    "BIRTH_CERT": ["Свидетельство о рождении: {BIRTH_CERT}.", "Выдан документ {BIRTH_CERT}."],
    "KLADR":      ["Код КЛАДР: {KLADR}.",             "Адрес по КЛАДР: {KLADR}."],
}

def gen_inn():
    while True:
        base = [random.randint(0,9) for _ in range(10)]
        c11 = [7,2,4,10,3,5,9,4,6,8]
        n11 = (sum(base[i]*c11[i] for i in range(10)) % 11) % 10
        d11 = base + [n11]
        c12 = [3,7,2,4,10,3,5,9,4,6,8]
        n12 = (sum(d11[i]*c12[i] for i in range(11)) % 11) % 10
        res = "".join(map(str, base + [n11, n12]))
        if validate_inn(res): return res

def gen_card():
    while True:
        body = "4" + "".join(str(random.randint(0,9)) for _ in range(14))
        for cd in range(10):
            cand = body + str(cd)
            if validate_luhn(cand): return cand

def gen_passport():
    region = random.choice(sorted(PASSPORT_REGION_WHITELIST))
    dept   = random.randint(0, 99)
    series = f"{region:02d}{dept:02d}"
    number = "".join(str(random.randint(0,9)) for _ in range(6))
    fmt = random.choice(["space","dash","compact"])
    if fmt == "space":   return f"{series[:2]} {series[2:]} {number}"
    if fmt == "dash":    return f"{series[:2]}-{series[2:]}-{number}"
    return series + number

def gen_ogrnip():
    while True:
        body = "3" + "".join(str(random.randint(0,9)) for _ in range(13))
        ctrl = (int(body) % 13) % 10
        res = body + str(ctrl)
        if validate_ogrnip(res): return res

def gen_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))

def gen_birth_cert():
    roman = ["I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII","XIII","XIV","XV"]
    series = random.choice(roman)
    num = "".join(str(random.randint(0,9)) for _ in range(6))
    fmt = random.choice(["plain","subj_space","subj_dash"])
    subj = random.choice(["АК","МЮ","МК","АБ","БА","ЮК"])
    if fmt == "plain":       return f"{series} {num}"
    if fmt == "subj_space":  return f"{series}-{subj} {num}"
    return f"{series}-{subj}-{num}"

def gen_kladr():
    region = random.choice(sorted(PASSPORT_REGION_WHITELIST))
    rest = "".join(str(random.randint(0,9)) for _ in range(11))
    return f"{region:02d}{rest}"

def gen_email():      return fake_ru.ascii_free_email()
def gen_name():       return fake_ru.first_name_male()
def gen_surname():    return fake_ru.last_name_male()

N_PER_ATTR = 60
test_texts = []

for _ in range(N_PER_ATTR):
    n, s = gen_name(), gen_surname()
    test_texts.append((random.choice(MANUAL_TEMPLATES["SURNAME"]).format(SURNAME=s, SURNAME_OBL=s+"у"), "SURNAME", s))
    test_texts.append((random.choice(MANUAL_TEMPLATES["NAME"]).format(NAME=n), "NAME", n))
    em = gen_email();      test_texts.append((random.choice(MANUAL_TEMPLATES["EMAIL"]).format(EMAIL=em), "EMAIL", em))
    inn = gen_inn();       test_texts.append((random.choice(MANUAL_TEMPLATES["INN"]).format(INN=inn), "INN", inn))
    cd = gen_card();       test_texts.append((random.choice(MANUAL_TEMPLATES["CREDIT_CARD"]).format(CARD=cd), "CREDIT_CARD", cd))
    pp = gen_passport();   test_texts.append((random.choice(MANUAL_TEMPLATES["PASSPORT"]).format(PASSPORT=pp), "PASSPORT", pp))
    og = gen_ogrnip();     test_texts.append((random.choice(MANUAL_TEMPLATES["OGRNIP"]).format(OGRNIP=og), "OGRNIP", og))
    ip = gen_ip();         test_texts.append((random.choice(MANUAL_TEMPLATES["IP"]).format(IP=ip), "IP", ip))
    bc = gen_birth_cert(); test_texts.append((random.choice(MANUAL_TEMPLATES["BIRTH_CERT"]).format(BIRTH_CERT=bc), "BIRTH_CERT", bc))
    kl = gen_kladr();      test_texts.append((random.choice(MANUAL_TEMPLATES["KLADR"]).format(KLADR=kl), "KLADR", kl))

from collections import Counter
cnt = Counter(a for _, a, _ in test_texts)
print("Количество Faker-тестов по атрибутам:", dict(cnt))
assert all(cnt[a] >= 50 for a in ATTRS), "Не хватает тестов!"

#%%
# ============================================================
# Ячейка 9. Оценка (factRuEval + Faker), weighted F1 и mean F1
# ============================================================
from collections import Counter

def compute_per_attr(predict_fn, samples):
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

def _find_span(text, value):
    idx = text.find(value)
    return (idx, idx + len(value)) if idx >= 0 else None

faker_samples = []
for text, attr, value in test_texts:
    sp = _find_span(text, value)
    if sp:
        faker_samples.append((text, [(attr, *sp)]))
cnt_faker = Counter(attr for _, gold in faker_samples for attr, _, _ in gold)

fact_samples = [(doc[0], doc[1]) for doc in factrueval_test_docs]

print(f"Faker test: {len(faker_samples)} текстов, по атрибутам: {dict(cnt_faker)}")
print(f"factRuEval test: {len(fact_samples)} документов")

per_attr_faker = compute_per_attr(predict, faker_samples)
per_attr_fact  = compute_per_attr(predict, fact_samples)
per_attr_total = {a: {k: per_attr_faker[a][k] + per_attr_fact[a][k]
                      for k in ("tp","fp","fn")} for a in ATTRS}

rows = metrics_from_per_attr(per_attr_total)
print(f"\n{'ATTR':<14}{'prec':>8}{'rec':>8}{'f1':>8}{'tp':>6}{'fp':>6}{'fn':>6}")
for a, p, r, f, tp, fp, fn in rows:
    print(f"{a:<14}{p:>8.3f}{r:>8.3f}{f:>8.3f}{tp:>6}{fp:>6}{fn:>6}")

weights = {a: per_attr_total[a]["tp"] + per_attr_total[a]["fn"] for a in ATTRS}
tw = sum(weights.values())
wp = sum(rows[i][1]*weights[ATTRS[i]] for i in range(len(ATTRS))) / tw
wr = sum(rows[i][2]*weights[ATTRS[i]] for i in range(len(ATTRS))) / tw
wf = sum(rows[i][3]*weights[ATTRS[i]] for i in range(len(ATTRS))) / tw
mean_f1 = sum(row[3] for row in rows) / len(ATTRS)

print(f"\nWeighted precision: {wp:.4f}")
print(f"Weighted recall   : {wr:.4f}")
print(f"Weighted F1       : {wf:.4f}")
print(f"Mean F1           : {mean_f1:.4f}")

def _all_checks():
    for _, attr, value in test_texts:
        if attr == "INN" and not validate_inn(value):
            raise AssertionError(f"Плохой ИНН: {value}")
        if attr == "CREDIT_CARD" and not validate_luhn(value):
            raise AssertionError(f"Плохая карта: {value}")
        if attr == "PASSPORT":
            series = re.sub(r"\D", "", value)[:4]
            if not validate_passport_series(series):
                raise AssertionError(f"Плохая серия: {series}")
        if attr == "OGRNIP" and not validate_ogrnip(value):
            raise AssertionError(f"Плохой ОГРНИП: {value}")
        if attr == "KLADR" and not validate_kladr(value):
            raise AssertionError(f"Плохой КЛАДР: {value}")
    return True
assert _all_checks()

print("\n--- Приёмочные проверки КМ2 ---")
assert len(fact_samples) > 0, "Не выполнен тест на factRuEval2016"
print(f"✅ Тест на factRuEval2016: {len(fact_samples)} документов")
assert all(cnt_faker[a] >= 50 for a in ATTRS), f"<50 тестов на атрибут: {dict(cnt_faker)}"
print(f"✅ Faker: {len(faker_samples)} текстов, ≥50 на каждый атрибут")
assert wf >= 0.85, f"Weighted F1 = {wf:.3f} < 0.85"
print(f"✅ Weighted F1 = {wf:.4f} ≥ 0.85")
assert mean_f1 >= 0.80, f"Mean F1 = {mean_f1:.3f} < 0.80"
print(f"✅ Mean F1 = {mean_f1:.4f} ≥ 0.80")
print("✅ Контрольные разряды валидны")
print("\n🎉 Все приёмочные условия КМ2 выполнены.")
# %%
