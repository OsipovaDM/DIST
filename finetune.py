"""Дообучение модели: не склеивать ФИО в одну сущность.

Запуск: python finetune_separate_fio.py [--epochs N] [--pairs M]

Логика:
    1. Бэкап текущей модели (один раз) в MODEL_OUT + "_backup".
    2. Загрузка factRuEval (devset).
    3. Генерация синтетических коротких ФИО через Faker (каждый запуск — новые).
    4. 1 эпоха дообучения на fact + augmented, с теми же гиперпараметрами.
    5. Сохранение обратно в MODEL_OUT.

При повторном запуске — ещё 1 эпоха с новой порцией синтетики.
"""
import argparse
import inspect
import os
import random
import shutil
import sys

# sys.path ДО импортов проекта — чтобы найти config/dataset/training
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluate
import numpy as np
from datasets import (
    Dataset as HFDataset, DatasetDict, Features,
    Sequence, Value, ClassLabel, concatenate_datasets,
)
from faker import Faker
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
)

from config import (
    MODEL_OUT, ATTRS, BERT_LABELS, BERT_LABEL2ID, BERT_ID2LABEL,
    MAX_LEN, SEED, VALID_SIZE, LR, BATCH_SIZE, WEIGHT_DECAY,
    WARMUP_RATIO, LR_SCHEDULER_TYPE, LOGGING_STEPS, SAVE_TOTAL_LIMIT,
    DATALOADER_NUM_WORKERS, REPORT_TO, METRIC_FOR_BEST_MODEL,
)
from dataset import load_factrueval

# ========== СХЕМА МЕТОК ==========
# Полная схема (21 метка) — на случай, если checkpoint обучен на всех атрибутах
ALL_LABELS = ["O"] + [f"{p}-{a}" for a in ATTRS for p in ("B", "I")]
ALL_LABEL2ID = {l: i for i, l in enumerate(ALL_LABELS)}
ALL_ID2LABEL = {i: l for l, i in ALL_LABEL2ID.items()}

# Runtime-схема: устанавливается в finetune() по факту num_labels модели
_LABELS = None
_LABEL2ID = None
_ID2LABEL = None

seqeval = evaluate.load("seqeval")

# ========== ПАРАМЕТРЫ ПО УМОЛЧАНИЮ ==========
NUM_EXTRA_EPOCHS = 1
N_AUG_PAIRS = 50     # 50 пар × 5 вариантов = 250 augmented примеров


# ========== АУГМЕНТАЦИЯ ==========
def _build_augmented_examples(n_pairs: int) -> list:
    """Синтетические короткие ФИО с правильной BIO-разметкой.

    Каждый запуск генерирует НОВЫЕ случайные имена (random без seed),
    поэтому модель видит разные примеры в каждом цикле дообучения.

    Варианты:
      1) Иван Петров                              → [B-NAME, B-SURNAME]
      2) Петров Иван                              → [B-SURNAME, B-NAME]
      3) Иванов Иван, Петров Пётр                 → [B-SURNAME, B-NAME, O, B-SURNAME, B-NAME]
      4) прокурор Иванов Иван выступил            → [O, B-SURNAME, B-NAME, O]
      5) прокурор Иванов Иван, депутат Петров Пётр → [O, B-SURNAME, B-NAME, O, O, B-SURNAME, B-NAME]
    """
    fake = Faker("ru_RU")
    rng = random.Random()     # без seed — каждый запуск уникальная выборка
    positions = ["прокурор", "депутат", "министр", "губернатор", "мэр", "судья"]
    out = []

    for _ in range(n_pairs):
        # --- случайные ФИО с согласованным полом ---
        is_male1 = rng.random() < 0.5
        name1 = fake.first_name_male() if is_male1 else fake.first_name_female()
        surname1 = fake.last_name_male() if is_male1 else fake.last_name_female()
        is_male2 = rng.random() < 0.5
        name2 = fake.first_name_male() if is_male2 else fake.first_name_female()
        surname2 = fake.last_name_male() if is_male2 else fake.last_name_female()
        pos1 = rng.choice(positions)
        pos2 = rng.choice(positions)

        # 1) Имя Фамилия (без контекста)
        out.append(([name1, surname1], ["B-NAME", "B-SURNAME"]))
        # 2) Фамилия Имя (без контекста) — паттерн, который сейчас ломается
        out.append(([surname1, name1], ["B-SURNAME", "B-NAME"]))
        # 3) Перечисление через запятую без контекста
        out.append((
            [surname1, name1, ",", surname2, name2],
            ["B-SURNAME", "B-NAME", "O", "B-SURNAME", "B-NAME"],
        ))
        # 4) Должность + ФИО (паттерн factRuEval)
        out.append((
            [pos1, surname1, name1, "выступил"],
            ["O", "B-SURNAME", "B-NAME", "O"],
        ))
        # 5) Должность + перечисление (реальный новостной паттерн)
        out.append((
            [pos1, surname1, name1, ",", pos2, surname2, name2],
            ["O", "B-SURNAME", "B-NAME", "O", "O", "B-SURNAME", "B-NAME"],
        ))

    return out


# ========== HF DATASET ==========
def to_hf(examples):
    """examples → HF Dataset с ClassLabel по текущей runtime-схеме."""
    data = {
        "tokens": [e[0] for e in examples],
        "ner_tags": [[_LABEL2ID[l] for l in e[1]] for e in examples],
    }
    features = Features({
        "tokens":   Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=_LABELS)),
    })
    return HFDataset.from_dict(data, features=features)


def _tokenize_and_align(batch, tokenizer):
    """Токенизация с выравниванием BIO по подсловам WordPiece."""
    enc = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True,
                    max_length=MAX_LEN)
    new_labels = []
    for i, labels in enumerate(batch["ner_tags"]):
        word_ids = enc.word_ids(batch_index=i)
        prev, seq = None, []
        for wid in word_ids:
            if wid is None:
                seq.append(-100)
            elif wid != prev:
                seq.append(labels[wid])
            else:
                lab = labels[wid]
                if _LABELS[lab].startswith("B-"):
                    seq.append(_LABEL2ID["I-" + _LABELS[lab][2:]])
                else:
                    seq.append(lab)
            prev = wid
        new_labels.append(seq)
    enc["labels"] = new_labels
    return enc


def compute_metrics(eval_pred):
    """precision / recall / F1 по BIO через seqeval."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    true_pred = [[_LABELS[p] for p, l in zip(pred, lab) if l != -100]
                 for pred, lab in zip(preds, labels)]
    true_lab  = [[_LABELS[l] for p, l in zip(pred, lab) if l != -100]
                 for pred, lab in zip(preds, labels)]
    res = seqeval.compute(predictions=true_pred, references=true_lab,
                          zero_division=0, scheme="IOB2")
    return {
        "precision": res["overall_precision"],
        "recall":    res["overall_recall"],
        "f1":        res["overall_f1"],
        "accuracy":  res["overall_accuracy"],
    }


# ========== ДООБУЧЕНИЕ ==========
def finetune(model_out: str = MODEL_OUT,
             num_epochs: int = NUM_EXTRA_EPOCHS,
             n_pairs: int = N_AUG_PAIRS) -> dict:
    """Одна эпоха дообучения существующей модели с аугментацией короткими ФИО."""
    global _LABELS, _LABEL2ID, _ID2LABEL

    if not os.path.exists(os.path.join(model_out, "config.json")):
        raise FileNotFoundError(f"Модель не найдена: {model_out}. Сначала обучите через main.py")

    # --- Бэкап модели (один раз) ---
    backup = model_out + "_backup"
    if not os.path.exists(backup):
        shutil.copytree(model_out, backup)
        print(f"Резервная копия: {backup}")

    # --- Загрузка данных ---
    print("\nЗагрузка factRuEval (devset)...")
    fact_examples, _ = load_factrueval()
    print(f"  factRuEval: {len(fact_examples)} документов")

    print(f"Генерация аугментации: {n_pairs} пар × 5 вариантов = {n_pairs * 5} примеров...")
    aug_examples = _build_augmented_examples(n_pairs)
    print(f"  augmented:  {len(aug_examples)} примеров")

    # --- Загрузка модели ДО установки схемы (нужно узнать num_labels) ---
    print(f"\nЗагрузка модели из {model_out}...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_out, ignore_mismatched_sizes=True,
    )
    n_labels = model.config.num_labels
    print(f"  num_labels в модели: {n_labels}")

    # --- Выбор схемы меток по факту num_labels ---
    if n_labels == len(ALL_LABELS):
        _LABELS, _LABEL2ID, _ID2LABEL = ALL_LABELS, ALL_LABEL2ID, ALL_ID2LABEL
        print(f"  Схема: ALL_LABELS ({n_labels} меток)")
    elif n_labels == len(BERT_LABELS):
        _LABELS, _LABEL2ID, _ID2LABEL = BERT_LABELS, BERT_LABEL2ID, BERT_ID2LABEL
        print(f"  Схема: BERT_LABELS ({n_labels} меток)")
    else:
        raise ValueError(f"Неизвестное число классов в модели: {n_labels}")

    # --- Сплит: fact → train+valid (80/20), augmented → train only ---
    fact_ds = to_hf(fact_examples).shuffle(seed=SEED)
    fact_split = fact_ds.train_test_split(test_size=VALID_SIZE, seed=SEED)
    aug_ds = to_hf(aug_examples)

    train_ds = concatenate_datasets([fact_split["train"], aug_ds])
    valid_ds = fact_split["test"]
    raw = DatasetDict({"train": train_ds, "valid": valid_ds})

    print(f"  train: {len(train_ds)} примеров (fact + aug)")
    print(f"  valid: {len(valid_ds)} примеров (только fact)")

    # --- Токенизация ---
    tokenizer = AutoTokenizer.from_pretrained(model_out)
    tokenizer.model_max_length = MAX_LEN
    tok_ds = raw.map(
        lambda batch: _tokenize_and_align(batch, tokenizer),
        batched=True,
        remove_columns=raw["train"].column_names,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # --- TrainingArguments (те же, что в training.py, но num_epochs=1) ---
    args_sig = set(inspect.signature(TrainingArguments.__init__).parameters)
    eval_key = "eval_strategy" if "eval_strategy" in args_sig else "evaluation_strategy"
    args_kwargs = {
        "output_dir": model_out,
        eval_key: "epoch",
        "save_strategy": "epoch",
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "learning_rate": LR,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "num_train_epochs": num_epochs,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "logging_steps": LOGGING_STEPS,
        "load_best_model_at_end": True,
        "metric_for_best_model": METRIC_FOR_BEST_MODEL,
        "greater_is_better": True,
        "report_to": REPORT_TO,
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,
        "seed": SEED,
    }
    if "save_safetensors" in args_sig:
        args_kwargs["save_safetensors"] = True
    training_args = TrainingArguments(**args_kwargs)

    # --- Trainer ---
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

    # --- Обучение и сохранение ---
    print(f"\nДообучение {num_epochs} эпох...")
    trainer.train()
    eval_metrics = trainer.evaluate()
    print(f"\nМетрики на valid (только factRuEval): {eval_metrics}")

    trainer.save_model(model_out)
    tokenizer.save_pretrained(model_out)
    print(f"Модель сохранена: {os.path.abspath(model_out)}")
    print(f"Бэкап предыдущей версии: {os.path.abspath(backup)}")

    # --- Проверка разделения ФИО (сброс кэша pipeline) ---
    print("\n=== Проверка: разделение ФИО ===")
    import recognize
    recognize._ner_pipe = None           # сбросить кэш — иначе загрузится старая модель
    recognize._tokenizer = None
    from recognize import predict
    for text in ["Иванов Иван", "Петров Иван", "Иванов Иван, Петров Пётр"]:
        print(f"  {text!r}: {predict(text)}")

    return eval_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Дообучение модели разделению ФИО")
    parser.add_argument("--epochs", type=int, default=NUM_EXTRA_EPOCHS,
                        help=f"Сколько эпох дообучать (по умолчанию {NUM_EXTRA_EPOCHS})")
    parser.add_argument("--pairs", type=int, default=N_AUG_PAIRS,
                        help=f"Сколько пар генерировать (× 5 вариантов, по умолчанию {N_AUG_PAIRS})")
    args = parser.parse_args()
    finetune(num_epochs=args.epochs, n_pairs=args.pairs)