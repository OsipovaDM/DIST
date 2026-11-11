"""Обучение BERT на размеченных данных"""
import inspect
import os

import evaluate
import numpy as np
import torch
from datasets import Dataset as HFDataset, DatasetDict, Features, Sequence, Value, ClassLabel
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,                                 
)

from config import (
    BASE_MODEL, NUM_EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY,
    WARMUP_RATIO, LR_SCHEDULER_TYPE, LOGGING_STEPS, SAVE_TOTAL_LIMIT,
    DATALOADER_NUM_WORKERS, REPORT_TO, METRIC_FOR_BEST_MODEL, VALID_SIZE,
    MAX_LEN, SEED,  BERT_LABELS, BERT_LABEL2ID, BERT_ID2LABEL, MODEL_OUT,
)

seqeval = evaluate.load("seqeval")

# ========== ОБЁРТКА examples → HF DATASET ==========
def to_hf(examples):
    """Список (tokens, BIO-labels) → HuggingFace Dataset с нужными типами.

    ClassLabel фиксирует порядок тегов — это гарантия, что id соответствуют
    LABELS при сохранении и не «съедут» при загрузке модели.
    """
    data = {
        "tokens":   [e[0] for e in examples],                 # список списков токенов
        "ner_tags": [[BERT_LABEL2ID[l] for l in e[1]] for e in examples],  # строковые теги → id
    }
    features = Features({
        "tokens":   Sequence(Value("string")),                # тип колонки: список строк
        "ner_tags": Sequence(ClassLabel(names=BERT_LABELS)),       # тип: список меток
    })
    return HFDataset.from_dict(data, features=features)       # HF Dataset


# ========== ТОКЕНИЗАЦИЯ И ВЫРАВНИВАНИЕ BIO ==========
def _tokenize_and_align(batch, tokenizer):
    """Токенизирует слова в подслова и выравнивает BIO-метки по подсловам.

    word_ids() возвращает номер исходного слова для каждого подслова.
    None → [CLS]/[SEP]/[PAD], метка -100 (Trainer игнорирует её в лоссе).
    Совпадает с предыдущим → это продолжение слова, B- заменяется на I-.
    Иначе → новое слово, метка копируется как есть.
    """
    enc = tokenizer(
        batch["tokens"],
        truncation=True,                                       # обрезаем до MAX_LEN
        is_split_into_words=True,                              # вход уже разбит на слова
        max_length=MAX_LEN,
    )
    all_labels = batch["ner_tags"]                             # id-шники BIO-тегов
    new_labels = []                                            # выровненные метки
    for i, labels in enumerate(all_labels):                    # по каждому примеру
        word_ids = enc.word_ids(batch_index=i)                 # номер слова для каждого подслова
        prev = None                                            # предыдущий word_id
        seq = []                                               # последовательность меток
        for wid in word_ids:
            if wid is None:                                    # спецтокен [CLS]/[SEP]/[PAD]
                seq.append(-100)                               # игнорируется в лоссе
            elif wid != prev:                                  # первый подслов нового слова
                seq.append(labels[wid])                        # берём тег исходного слова
            else:                                              # продолжение того же слова
                lab = labels[wid]                              # тег родительского слова
                if BERT_LABELS[lab].startswith("B-"):               # B- на первом подслове
                    seq.append(BERT_LABEL2ID["I-" + BERT_LABELS[lab][2:]])  # на продолжении — I-
                else:
                    seq.append(lab)                            # уже I- или O — оставляем
            prev = wid
        new_labels.append(seq)
    enc["labels"] = new_labels                                 # добавляем колонку labels
    return enc


# ========== МЕТРИКИ ДЛЯ TRAINER ==========
def compute_metrics(eval_pred):
    """precision / recall / F1 по BIO-тегам через seqeval.

    Trainer передаёт (logits, labels). Берём argmax по логитам,
    отбрасываем служебные токены (-100), считаем по схеме IOB2.
    """
    logits, labels = eval_pred                                # предсказания модели и истина
    preds = np.argmax(logits, axis=-1)                        # id предсказанных тегов
    true_pred = [[BERT_LABELS[p] for p, l in zip(pred, lab) if l != -100]
                 for pred, lab in zip(preds, labels)]         # предсказания без -100
    true_lab  = [[BERT_LABELS[l] for p, l in zip(pred, lab) if l != -100]
                 for pred, lab in zip(preds, labels)]         # истина без -100
    res = seqeval().compute(
        predictions=true_pred, references=true_lab,
        zero_division=0, scheme="IOB2",                       # стандартная схема BIO
    )
    return {
        "precision": res["overall_precision"],
        "recall":    res["overall_recall"],
        "f1":        res["overall_f1"],
        "accuracy":  res["overall_accuracy"],
    }


# ========== ОБУЧЕНИЕ МОДЕЛИ ==========
def train_model(examples, model_out=MODEL_OUT):
    """Обучает BERT на examples и сохраняет в model_out.

    examples  — список (tokens, BIO-labels), полученный из dataset.load_factrueval();
    model_out — папка, куда сохранить веса и токенизатор (по умолчанию MODEL_OUT).

    Возвращает dict с метриками лучшей эпохи (по eval_f1).
    Переобучение с нуля — задача main.py (удалить папку и вызвать train_model).
    """
    os.makedirs(os.path.dirname(model_out), exist_ok=True)    # создаём родительскую папку модели

    # --- Подготовка train/valid ---
    full_ds = to_hf(examples).shuffle(seed=SEED)              # HF Dataset + перемешивание
    split = full_ds.train_test_split(test_size=VALID_SIZE, seed=SEED)  # 80/20
    raw = DatasetDict({"train": split["train"], "valid": split["test"]})  # две колонки

    # --- Токенизация ---
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)     # токенизатор предобученной модели
    tok_ds = raw.map(
        lambda batch: _tokenize_and_align(batch, tokenizer),   # замыкание с токенизатором
        batched=True,                                         # батчами быстрее
        remove_columns=raw["train"].column_names,              # убираем tokens/ner_tags
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)  # паддинг батчей

    # --- Версионно-независимый TrainingArguments ---
    args_sig = set(inspect.signature(TrainingArguments.__init__).parameters)  # какие kwargs знает версия
    eval_key = "eval_strategy" if "eval_strategy" in args_sig else "evaluation_strategy"  # имя параметра для стратегии eval
    args_kwargs = {
        "output_dir": model_out,                              # куда сохранять чекпоинты
        eval_key: "epoch",                                    # оценивать в конце каждой эпохи
        "save_strategy": "epoch",                             # сохранять чекпоинт каждую эпоху
        "save_total_limit": SAVE_TOTAL_LIMIT,                 # хранить не больше N чекпоинтов
        "learning_rate": LR,                                  # скорость обучения
        "per_device_train_batch_size": BATCH_SIZE,            # размер батча при обучении
        "per_device_eval_batch_size": BATCH_SIZE,             # размер батча при валидации
        "num_train_epochs": NUM_EPOCHS,                       # число эпох
        "weight_decay": WEIGHT_DECAY,                         # L2-регуляризация
        "warmup_ratio": WARMUP_RATIO,                         # доля шагов на разогрев LR
        "lr_scheduler_type": LR_SCHEDULER_TYPE,               # тип затухания LR
        "logging_steps": LOGGING_STEPS,                       # как часто печатать loss
        "load_best_model_at_end": True,                       # откатиться к лучшей эпохе
        "metric_for_best_model": METRIC_FOR_BEST_MODEL,       # по какой метрике выбирать лучшую
        "greater_is_better": True,                            # для F1 больше = лучше
        "report_to": REPORT_TO,                               # не слать логи во внешние сервисы
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,     # воркеры DataLoader (0 = основной поток)
        "seed": SEED,                                         # воспроизводимость
    }
    if "save_safetensors" in args_sig:                        # параметр появился в 4.35+
        args_kwargs["save_safetensors"] = True
    training_args = TrainingArguments(**args_kwargs)

    # --- Модель ---
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(BERT_LABELS),                               # голова под наше число классов
        id2label=BERT_ID2LABEL,                                    # id → строка тега
        label2id=BERT_LABEL2ID,                                    # строка → id
        ignore_mismatched_sizes=True,                         # голова инициализируется заново
    )

    # --- Trainer (версионно-независимый аргумент токенизатора) ---
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
    trainer.train()                                           # обучение; лучшая эпоха откатится автоматически
    eval_metrics = trainer.evaluate()                         # метрики лучшей эпохи
    trainer.save_model(model_out)                             # веса
    tokenizer.save_pretrained(model_out)                      # токенизатор + конфиг
    return eval_metrics                                       # dict для печати в main.py