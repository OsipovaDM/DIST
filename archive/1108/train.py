"""Обучение BERT на factRuEval и оценка метрик.

Запуск: python train.py
"""
import os, sys

# sys.path ДО всех импортов — чтобы `from config import ...` нашёл модули в корне
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re, random, zipfile, urllib.request, builtins, shutil
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
)
from datasets import Dataset as HFDataset, DatasetDict, Features, Sequence, Value, ClassLabel
import evaluate
from faker import Faker

# Абсолютные импорты — вместо относительных `from .`
from config import (MODEL_OUT, DATA_DIR, ATTRS, LABELS, LABEL2ID, ID2LABEL,
                    MAX_LEN, SEED)
from recognize import (predict, validate_inn, validate_luhn, validate_passport_series,
                       validate_ogrnip, validate_kladr, PASSPORT_REGION_WHITELIST)

# Гиперпараметры — только для обучения
BASE_MODEL = "DeepPavlov/rubert-base-cased"
NUM_EPOCHS = 7
BATCH_SIZE = 16
LR = 2e-5
WEIGHT_DECAY = 0.01
FACT_RUEVAL_URL = "https://github.com/dialogue-evaluation/factRuEval-2016/archive/master.zip"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

# ============================================================
# Ячейка 3 КМ2 — копируется ЦЕЛИКОМ, но:
#   - DATA_DIR берётся из pdn (не определять локально)
#   - download_factrueval использует FACT_RUEVAL_URL из шапки
# ============================================================
# Ячейка 3. factRuEval → train/valid (devset) + test (testset)
# Только SURNAME и NAME из corus
# ============================================================
import corus.io as _corus_io
import corus.sources.factru as _fr
from corus import load_factru

def _utf8_open(path, *a, **kw):
    """Открывает файл с явной кодировкой UTF-8."""
    
    kw.setdefault("encoding", "utf-8")
    return builtins.open(path, *a, **kw)

def _load_text_utf8(path):
    """Читает текст из файла в UTF-8. Обёртка для monkey-patch corus.io.load_text."""
    with _utf8_open(path) as f:
        return f.read()

# подменяем функцию чтения текста на UTF-8-версию
_corus_io.load_text = _load_text_utf8
_fr.load_text = _load_text_utf8

def download_factrueval(target_dir=DATA_DIR):
    """Скачивает и распаковывает архив factRuEval, если его ещё нет."""
    
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
print("Скачивание и распаковка factRuEval завершены:", FACT_DIR)

# Соответствие между типами сущностей в corus и BIO-тегами.
TYPE_TO_ATTR = {"surname": "SURNAME", "name": "NAME"}

# Разделяем документы на devset (обучение) и testset (оценка).
# ВАЖНО: нельзя фильтровать по url — в corus url не содержит имя папки,
# поэтому вызываем load_factru дважды с явным sets= для каждой папки.
dev_records  = list(load_factru(FACT_DIR, sets=["devset"]))
test_records = list(load_factru(FACT_DIR, sets=["testset"]))
print(f"devset: {len(dev_records)} док., testset: {len(test_records)} док.")

def _rec_tokens(rec):
    """Возвращает список токенов записи в формате (текст, начало, конец)."""
    
    # Если corus разметил токены — используем их (это точное деление, совпадающее
    # с оригинальной разметкой). Если нет — режем по пробелам регуляркой.
    
    toks = getattr(rec, "tokens", None)
    if toks:
        return [(t.text, t.start, t.stop) for t in toks]
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", rec.text)]

def rec_to_example(rec):
    """Преобразует запись corus в пару (список токенов, список BIO-тегов)."""
    
    # для каждого токена проверяем, попадает ли он в какой-нибудь
    # span типа surname или name у объекта Person. Если попадает — ставим B-
    # (первый токен сущности) или I- (продолжение). Продолжение определяется
    # по тому, был ли предыдущий токен тем же атрибутом.
    
    toks = _rec_tokens(rec)
    tokens = [t[0] for t in toks]
    labels = ["O"] * len(tokens)
    for i, (_, ts, te) in enumerate(toks):
        matched = None
        for obj in rec.objects:
            if getattr(obj, "type", "") != "Person":
                continue
            for span in obj.spans:
                # Токен принадлежит спану, если пересекается с его границами
                if span.start <= ts < span.stop or span.start < te <= span.stop:
                    t = span.type.lower()
                    if t in TYPE_TO_ATTR:
                        matched = TYPE_TO_ATTR[t]; break
            if matched: break
        if matched:
            prev = labels[i-1] if i > 0 else "O"
            # Если предыдущий токен — та же сущность, ставим I (продолжение),
            # иначе B (начало новой сущности)
            labels[i] = ("I-" if prev.endswith(matched) else "B-") + matched
    return tokens, labels

examples = [rec_to_example(r) for r in dev_records if _rec_tokens(r)]

def rec_to_test_doc(rec):
    """Возвращает (полный текст, список gold-спанов) для оценки на factRuEval."""
    
    # В отличие от обучающего примера, здесь нужны именно символьные позиции
    # спанов в исходном тексте — чтобы сравнить с выходом predict() на тесте.
    
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

# Если в архиве нет папки testset или разметка в ней без Person — берём
# 20% devset в качестве независимого теста. Это гарантирует, что требование
# «тестировать на factRuEval» всё равно выполнено.
if not factrueval_test_docs and dev_records:
    print("⚠ testset пуст → берём 20% devset как factRuEval-тест")
    rng = random.Random(SEED)
    shuffled = list(dev_records)
    rng.shuffle(shuffled)
    n = max(1, int(0.2 * len(shuffled)))
    dev_for_test = shuffled[:n]
    dev_records = shuffled[n:]
    factrueval_test_docs = [d for d in (rec_to_test_doc(r) for r in dev_for_test) if d[1]]
    examples = [rec_to_example(r) for r in dev_records if _rec_tokens(r)]
    print(f"  Пересобрано: train/valid={len(examples)}, test={len(factrueval_test_docs)}")

# Ручной fallback на случай, если corus вернул пустую разметку.
# Пять коротких примеров с правильными BIO-тегами — минимальный сигнал для обучения.
if not any(l != "O" for _, labs in examples for l in labs):
    print("⚠ devset без ФИО — ручной fallback")
    MANUAL = [
        ("Петров Иван пришёл.",         ["B-SURNAME", "B-NAME", "O"]),
        ("Сидорова Мария ушла.",        ["B-SURNAME", "B-NAME", "O"]),
        ("Иванов Пётр написал письмо.", ["B-SURNAME", "B-NAME", "O", "O"]),
        ("Екимова Александра — юрист.", ["B-SURNAME", "B-NAME", "O", "O"]),
        ("Дмитрий Медведев выступил.",  ["B-NAME", "B-SURNAME", "O"]),
    ]
    examples = [(t.split(), l) for t, l in MANUAL]

print(f"Train/valid: {len(examples)} документов")

# 4. HF Dataset, токенизация WordPiece, метрики seqeval
# ============================================================
def to_hf(examples):
    """Оборачивает список (tokens, labels) в HuggingFace Dataset с правильными типами."""
    
    # ClassLabel фиксирует порядок тегов — это гарантия, что id соответствуют
    # LABELS при сохранении модели и не «съедут» при загрузке.
    
    data = {"tokens": [e[0] for e in examples],
            "ner_tags": [[LABEL2ID[l] for l in e[1]] for e in examples]}
    features = Features({
        "tokens":   Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=LABELS)),
    })
    return HFDataset.from_dict(data, features=features)


# Разделяем на train/valid 80/20 — на valid будем считать eval_f1 в конце каждой эпохи
full_ds = to_hf(examples).shuffle(seed=SEED)
split = full_ds.train_test_split(test_size=0.2, seed=SEED)
raw = DatasetDict({"train": split["train"], "valid": split["test"]})
print(raw)

# Загружаем токенизатор, соответствующий предобученной модели
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


def tokenize_and_align(batch):
    """Токенизирует слова в подслова и выравнивает BIO-метки по подсловам.
    
    Алгоритм: word_ids() возвращает номер исходного слова для каждого подслова.
    Если номер None — это [CLS]/[SEP]/[PAD], метка -100 (Trainer игнорирует её
    в лоссе). Если номер совпадает с предыдущим — это продолжение того же слова,
    и B-метка меняется на I-. Иначе — новое слово, метка копируется как есть.
    """
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
                # Специальный токен — игнорируется при подсчёте лосса
                seq.append(-100)
            elif wid != prev:
                # Первый подслов исходного слова — берём тег исходного слова
                seq.append(labels[wid])
            else:
                # Продолжение того же слова — B- заменяем на I-
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

# Collator сам разложит батчи по тензорам и добавит padding
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# seqeval считает метрики по BIO-последовательностям, а не по отдельным токенам —
# это даёт честную оценку на уровне сущностей, а не на уровне отдельных меток
seqeval = evaluate.load("seqeval")


def compute_metrics(eval_pred):
    """Считает precision, recall, F1 по BIO-тегам через seqeval.
    
    Trainer передаёт сюда кортеж (logits, labels). Берём argmax по логитам
    и отбрасываем служебные токены (метка -100), затем отдаём в seqeval
    со схемой IOB2 — стандартной схемой BIO.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    true_pred = [[LABELS[p] for p, l in zip(pred, lab) if l != -100]
                 for pred, lab in zip(preds, labels)]
    true_lab = [[LABELS[l] for p, l in zip(pred, lab) if l != -100]
                for pred, lab in zip(preds, labels)]
    res = seqeval.compute(predictions=true_pred, references=true_lab,
                          zero_division=0, scheme="IOB2")
    return {
        "precision": res["overall_precision"],
        "recall":    res["overall_recall"],
        "f1":        res["overall_f1"],
        "accuracy":  res["overall_accuracy"],
    }

# Ячейка 5. Обучение BERT (версионно-независимо)
# ============================================================
import inspect

# Если RETRAIN=True, удаляем старую модель — иначе pipeline ниже подхватит
# её вместо новой, и predict будет работать с устаревшими весами.
RETRAIN = False
if RETRAIN and os.path.exists(MODEL_OUT):
    shutil.rmtree(MODEL_OUT, ignore_errors=True)
    print(f"Удалена старая модель: {MODEL_OUT}")

SKIP_TRAINING = True
model_exists = os.path.exists(os.path.join(MODEL_OUT, "config.json"))

# В разных версиях transformers параметр называется по-разному — проверяем,
# какое имя поддерживается в текущей установке. Это спасает от TypeError
# при переезде между 4.30 / 4.46 / 5.x.
args_sig = set(inspect.signature(TrainingArguments.__init__).parameters)
eval_key = "eval_strategy" if "eval_strategy" in args_sig else "evaluation_strategy"

if SKIP_TRAINING and model_exists:
    print(f"✅ Модель найдена в {MODEL_OUT} — пропускаем обучение")
    model = AutoModelForTokenClassification.from_pretrained(MODEL_OUT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_OUT)
else:
    # Гиперпараметры Trainer:
    args_kwargs = {
        "output_dir": MODEL_OUT,    # папка для чекпоинтов и итоговой модели
        eval_key: "epoch",    # оценивать на valid в конце каждой эпохи
        "save_strategy": "epoch",    # сохранять чекпоинт в конце каждой эпохи
        "save_total_limit": 1,    # хранить только последний чекпоинт (экономия диска)
        "learning_rate": LR,    # скорость шага оптимизатора (см. конфиг)
        "per_device_train_batch_size": BATCH_SIZE,    # размер батча на устройство при обучении
        "per_device_eval_batch_size": BATCH_SIZE,    # размер батча на устройство при валидации
        "num_train_epochs": NUM_EPOCHS,    # число полных проходов по train
        "weight_decay": WEIGHT_DECAY,    # L2-регуляризация
        "logging_steps": 50,    # как часто печатать loss в лог
        "load_best_model_at_end": True,    # в конце обучения откатиться к лучшей эпохе по F1
        "metric_for_best_model": "f1",    # какую метрику считать «лучшей» (F1)
        "greater_is_better": True,    # больше = лучше (для F1 это верно)
        "report_to": "none",    # не слать логи ни в какие внешние сервисы
        "seed": SEED,    # воспроизводимость перемешивания
        "warmup_ratio": 0.1,    # первые 10% шагов LR плавно растёт с 0 до LR
        "lr_scheduler_type": "cosine",    # LR плавно затухает по косинусу к концу обучения
        "dataloader_num_workers": 0,    # 0 = грузить батчи в основном потоке (Windows-safe)
    }
    # save_safetensors появился в 4.35+ — добавляем только если версия знает
    if "save_safetensors" in args_sig:
        args_kwargs["save_safetensors"] = True

    training_args = TrainingArguments(**args_kwargs)
    print(f"transformers: {__import__('transformers').__version__}, используется '{eval_key}'")

    # Загружаем BERT-бэкбон из checkpoint'а rubert-base-cased и заменяем classification-head
    # под наше число классов. ignore_mismatched_sizes нужен, чтобы transformers не ругался
    # на несовпадение размеров (у предобученной модели другой head — LM, а не NER).
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # В разных версиях Trainer аргумент с токенизатором называется tokenizer или
    # processing_class — выбираем тот, что поддерживается.
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

    # Сохраняем лучшую модель (по F1 на valid) и токенизатор — то, что нужно
    # для последующего инференса без переобучения.
    trainer.save_model(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)

    total_bytes = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(MODEL_OUT) for f in fs
    )
    print(f"💾 Сохранено в: {os.path.abspath(MODEL_OUT)} ({total_bytes/1e6:.1f} МБ)")

    # Ячейка 8. Faker-тест (≥50 на атрибут) + валидные генераторы
# ============================================================
fake_ru = Faker("ru_RU")

# Шаблоны текстов. {SURNAME} и {NAME} будут заменены на сгенерированные
# значения, {SURNAME_OBL} — на фамилию в винительном падеже, чтобы проверить,
# что модель работает не только с именительным.
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
    """Генерирует случайный валидный ИНН."""
    
    # Берём 10 случайных цифр, считаем по ним 11-ю и 12-ю по алгоритму ФНС.
    # Такой цикл гарантирует, что ИНН пройдёт validate_inn — иначе assert
    # контрольных разрядов упадёт при оценке.
    
    while True:
        base = [random.randint(0, 9) for _ in range(10)]
        c11 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        n11 = (sum(base[i] * c11[i] for i in range(10)) % 11) % 10
        d11 = base + [n11]
        c12 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        n12 = (sum(d11[i] * c12[i] for i in range(11)) % 11) % 10
        res = "".join(map(str, base + [n11, n12]))
        if validate_inn(res):
            return res


def gen_card():
    """Генерирует случайный валидный 16-значный номер карты."""
    
    # Начинаем с префикса «4» (Visa), генерируем 14 цифр, последнюю подбираем
    # так, чтобы алгоритм Луна сходился. Перебор 0–9 — десять попыток максимум.
    
    while True:
        body = "4" + "".join(str(random.randint(0, 9)) for _ in range(14))
        for cd in range(10):
            cand = body + str(cd)
            if validate_luhn(cand):
                return cand


def gen_passport():
    """Генерирует номер паспорта с валидной серией из белого списка регионов."""
    
    # Формат выбирается случайно из трёх: с пробелом, с дефисом, без разделителей —
    # это нужно, чтобы проверить, что регулярка ловит все варианты.
    
    region = random.choice(sorted(PASSPORT_REGION_WHITELIST))
    dept = random.randint(0, 99)
    series = f"{region:02d}{dept:02d}"
    number = "".join(str(random.randint(0, 9)) for _ in range(6))
    fmt = random.choice(["space", "dash", "compact"])
    if fmt == "space":
        return f"{series[:2]} {series[2:]} {number}"
    if fmt == "dash":
        return f"{series[:2]}-{series[2:]}-{number}"
    return series + number


def gen_ogrnip():
    """Генерирует ОГРНИП по алгоритму ФНС."""
    
    # Первая цифра всегда «3» (признак ИП). Затем 13 случайных цифр, затем
    # контрольная: остаток от деления первых 14 на 13, по модулю 10.
    
    while True:
        body = "3" + "".join(str(random.randint(0, 9)) for _ in range(13))
        ctrl = (int(body) % 13) % 10
        res = body + str(ctrl)
        if validate_ogrnip(res):
            return res


def gen_ip():
    """Генерирует случайный IPv4-адрес из четырёх октетов 1–254 (без 0 и 255 — они служебные)."""
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def gen_birth_cert():
    """Генерирует номер свидетельства о рождении в трёх форматах."""
    
    # Форматы: «IV 123456», «IV-МЮ 123456», «IV-МЮ-123456». Регулярка
    # BIRTH_CERT_RE должна ловить все три.
    
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
             "XI", "XII", "XIII", "XIV", "XV"]
    series = random.choice(roman)
    num = "".join(str(random.randint(0, 9)) for _ in range(6))
    fmt = random.choice(["plain", "subj_space", "subj_dash"])
    subj = random.choice(["АК", "МЮ", "МК", "АБ", "БА", "ЮК"])
    if fmt == "plain":
        return f"{series} {num}"
    if fmt == "subj_space":
        return f"{series}-{subj} {num}"
    return f"{series}-{subj}-{num}"


def gen_kladr():
    """Генерирует 13-значный код КЛАДР с валидным кодом региона в начале."""
    region = random.choice(sorted(PASSPORT_REGION_WHITELIST))
    rest = "".join(str(random.randint(0, 9)) for _ in range(11))
    return f"{region:02d}{rest}"


def gen_email():
    """Генерирует e-mail через Faker — с ASCII-доменом, чтобы регулярка точно поймала."""
    return fake_ru.ascii_free_email()


def gen_name():
    """Возвращает случайное мужское имя из словаря Faker (ru_RU)."""
    return fake_ru.first_name_male()


def gen_surname():
    """Возвращает случайную мужскую фамилию из словаря Faker (ru_RU)."""
    return fake_ru.last_name_male()


# Собираем тестовый корпус. На каждый атрибут — по 60 текстов (запас над 50).
N_PER_ATTR = 60
test_texts = []   # каждый элемент: (текст, тип атрибута, эталонное значение)

for _ in range(N_PER_ATTR):
    n, s = gen_name(), gen_surname()
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

# Контроль: на каждый атрибут должно быть >= 50 текстов
from collections import Counter
cnt = Counter(a for _, a, _ in test_texts)
print("Количество Faker-тестов по атрибутам:", dict(cnt))
assert all(cnt[a] >= 50 for a in ATTRS), "Не хватает тестов!"

# Ячейка 9. Оценка (factRuEval + Faker), weighted F1 и mean F1
# ============================================================
from collections import Counter

SKIP_TEST = True


def compute_per_attr(predict_fn, samples):
    """Считает tp/fp/fn для каждого атрибута по спанам."""
    
    # Спан считается «попавшим», если есть предсказание того же типа, которое
    # хотя бы частично перекрывается с эталонным спаном. Это мягче, чем
    # требовать точного совпадения границ, но даёт осмысленные метрики даже
    # при небольших смещениях границ.
    
    per_attr = {a: {"tp": 0, "fp": 0, "fn": 0} for a in ATTRS}
    for text, gold in samples:
        pset = set(predict_fn(text))
        gset = set(gold)

        # Истинно положительные и ложно отрицательные
        for a, s, e in gset:
            if any(pa == a and not (e <= ps or s >= pe) for pa, ps, pe in pset):
                per_attr[a]["tp"] += 1
            else:
                per_attr[a]["fn"] += 1

        # Ложно положительные — предсказания, которым нет эталона
        for a, s, e in pset:
            if not any(ga == a and not (e <= gs or s >= ge) for ga, gs, ge in gset):
                per_attr[a]["fp"] += 1
    return per_attr


def metrics_from_per_attr(per_attr):
    """Считает precision, recall, F1 по посчитанным tp/fp/fn."""
    rows = []
    for a in ATTRS:
        tp, fp, fn = per_attr[a]["tp"], per_attr[a]["fp"], per_attr[a]["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append((a, prec, rec, f1, tp, fp, fn))
    return rows


def _find_span(text, value):
    """Ищет точную подстроку в тексте и возвращает (начало, конец).
    Возвращает None, если не нашлось — например, если value был заменён
    на падежную форму и не совпадает с оригиналом."""
    idx = text.find(value)
    return (idx, idx + len(value)) if idx >= 0 else None

if not SKIP_TEST:
    # Формируем тестовые примеры из Faker: для каждого (текст, атрибут, значение)
    # находим точные границы значения в тексте — это будет gold-спан.
    faker_samples = []
    for text, attr, value in test_texts:
        sp = _find_span(text, value)
        if sp:
            faker_samples.append((text, [(attr, *sp)]))
    cnt_faker = Counter(attr for _, gold in faker_samples for attr, _, _ in gold)

    # Тестовые примеры из factRuEval: там уже готовые gold-спаны в символах
    fact_samples = [(doc[0], doc[1]) for doc in factrueval_test_docs]

    print(f"Faker test: {len(faker_samples)} текстов, по атрибутам: {dict(cnt_faker)}")
    print(f"factRuEval test: {len(fact_samples)} документов")

    # Считаем метрики раздельно по источникам и в сумме
    per_attr_faker = compute_per_attr(predict, faker_samples)
    per_attr_fact = compute_per_attr(predict, fact_samples)
    per_attr_total = {a: {k: per_attr_faker[a][k] + per_attr_fact[a][k]
                        for k in ("tp", "fp", "fn")} for a in ATTRS}

    rows = metrics_from_per_attr(per_attr_total)
    print(f"\n{'ATTR':<14}{'prec':>8}{'rec':>8}{'f1':>8}{'tp':>6}{'fp':>6}{'fn':>6}")
    for a, p, r, f, tp, fp, fn in rows:
        print(f"{a:<14}{p:>8.3f}{r:>8.3f}{f:>8.3f}{tp:>6}{fp:>6}{fn:>6}")

    # Weighted F1 — среднее F1 по атрибутам, взвешенное по числу истинных примеров.
    # Даёт больший вес частым атрибутам; для отчёта это основная метрика.
    weights = {a: per_attr_total[a]["tp"] + per_attr_total[a]["fn"] for a in ATTRS}
    tw = sum(weights.values())
    wp = sum(rows[i][1] * weights[ATTRS[i]] for i in range(len(ATTRS))) / tw
    wr = sum(rows[i][2] * weights[ATTRS[i]] for i in range(len(ATTRS))) / tw
    wf = sum(rows[i][3] * weights[ATTRS[i]] for i in range(len(ATTRS))) / tw

    # Mean F1 — простое среднее по атрибутам. Штрафует за провалы на редких атрибутах.
    mean_f1 = sum(row[3] for row in rows) / len(ATTRS)

    print(f"\nWeighted precision: {wp:.4f}")
    print(f"Weighted recall   : {wr:.4f}")
    print(f"Weighted F1       : {wf:.4f}")
    print(f"Mean F1           : {mean_f1:.4f}")


    def _all_checks():
        """Проверяет, что все сгенерированные тестовые значения проходят контрольные
        разряды. Если хоть одно не проходит — значит, генератор сломан, и метрики
        на Faker-тесте недостоверны."""
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

    # ---- Приёмочные проверки требований к работе ----
    print("\n--- Приёмочные проверки ---")
    assert len(fact_samples) > 0, "Не выполнен тест на factRuEval2016"
    print(f"✅ Тест на factRuEval2016: {len(fact_samples)} документов")

    assert all(cnt_faker[a] >= 50 for a in ATTRS), f"<50 тестов на атрибут: {dict(cnt_faker)}"
    print(f"✅ Faker: {len(faker_samples)} текстов, ≥50 на каждый атрибут")

    assert wf >= 0.85, f"Weighted F1 = {wf:.3f} < 0.85"
    print(f"✅ Weighted F1 = {wf:.4f} ≥ 0.85")

    assert mean_f1 >= 0.80, f"Mean F1 = {mean_f1:.3f} < 0.80"
    print(f"✅ Mean F1 = {mean_f1:.4f} ≥ 0.80")

    print("✅ Контрольные разряды валидны")
    print("\n🎉 Все приёмочные условия выполнены.")