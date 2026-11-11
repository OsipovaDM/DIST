"""Оркестратор проекта: обучение, оценка, демонстрация.

    1. Если RETRAIN=True — удалить старую модель и обучить заново.
    3. Если SKIP_TEST=False — сгенерировать тесты, посчитать метрики, проверить пороги.
    4. Всегда — демонстрация распознавания, обезличивания и сквозного цикла.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_OUT, RETRAIN, SKIP_TEST,
    THRESHOLD_WEIGHTED_F1, THRESHOLD_MEAN_F1, MIN_TEST_DOCS, N_PER_ATTR,
)


# ========== ОБУЧЕНИЕ ==========
def run_training(examples) -> dict:
    """Обучает BERT на examples, возвращает метрики валидации."""
    from training import train_model                         # импорт здесь — чтобы не тянуть transformers зря
    print(f"Обучение BERT на {len(examples)} документах...")
    metrics = train_model(examples, MODEL_OUT)               # обучение + сохранение
    print(f"✅ Обучение завершено. Метрики на valid: {metrics}")
    print(f"Модель сохранена в {os.path.abspath(MODEL_OUT)}")
    return metrics


# ========== ОЦЕНКА ==========
def run_evaluation(factrueval_test_docs) -> dict:
    """Считает метрики на Faker + factRuEval, проверяет приёмочные пороги."""
    from evaluation import evaluate                          # импорт здесь — не тянет faker при import main

    print(f"\nОценка на Faker (≥{N_PER_ATTR} на атрибут) + factRuEval ({len(factrueval_test_docs)} док.)...")
    result = evaluate(factrueval_test_docs, n_per_attr=N_PER_ATTR)  # оценка

    print(f"\n{'ATTR':<14}{'prec':>8}{'rec':>8}{'f1':>8}{'tp':>6}{'fp':>6}{'fn':>6}")
    for a, p, r, f, tp, fp, fn in result["rows"]:             # печать таблицы по атрибутам
        print(f"{a:<14}{p:>8.3f}{r:>8.3f}{f:>8.3f}{tp:>6}{fp:>6}{fn:>6}")

    print(f"\nWeighted F1: {result['weighted_f1']:.4f}")
    print(f"Mean F1:     {result['mean_f1']:.4f}")
    print(f"Faker:       {result['n_faker']} текстов, по атрибутам: {result['cnt_faker']}")
    print(f"factRuEval:  {result['n_fact']} документов")

    # --- Приёмочные проверки (пороги из config) ---
    print("\n--- Приёмочные проверки ---")
    assert result["valid_data"], "Не все сгенерированные значения прошли контрольные разряды"
    print("✅ Контрольные разряды валидны")

    assert result["n_fact"] >= MIN_TEST_DOCS, f"Мало документов в factRuEval-тесте: {result['n_fact']}"
    print(f"✅ factRuEval-тест: {result['n_fact']} документов (≥{MIN_TEST_DOCS})")

    assert all(v >= N_PER_ATTR for v in result["cnt_faker"].values()), \
        f"Меньше {N_PER_ATTR} текстов на атрибут: {result['cnt_faker']}"
    print(f"✅ Faker: ≥{N_PER_ATTR} текстов на каждый атрибут")

    assert result["weighted_f1"] >= THRESHOLD_WEIGHTED_F1, \
        f"Weighted F1 = {result['weighted_f1']:.4f} < {THRESHOLD_WEIGHTED_F1}"
    print(f"✅ Weighted F1 = {result['weighted_f1']:.4f} ≥ {THRESHOLD_WEIGHTED_F1}")

    assert result["mean_f1"] >= THRESHOLD_MEAN_F1, \
        f"Mean F1 = {result['mean_f1']:.4f} < {THRESHOLD_MEAN_F1}"
    print(f"✅ Mean F1 = {result['mean_f1']:.4f} ≥ {THRESHOLD_MEAN_F1}")

    print("\nВсе приёмочные условия выполнены.")
    return result


# ========== ДЕМОНСТРАЦИЯ РАСПОЗНАВАНИЯ ==========
def demo_recognition():
    """Показывает распознавание на одном тексте со всеми атрибутами и форматами.

    В DEMO_TEXT встречается каждый из 10 атрибутов во всех поддерживаемых
    форматах: слитно, через пробел, через дефис, смешанный.
    """
    from collections import defaultdict
    from recognize import predict
    from config import DEMO_TEXT, ATTRS

    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ: распознавание ПДн (все атрибуты, все форматы)")
    print("=" * 60)
    print(f"\nИсходный текст ({len(DEMO_TEXT)} символов):\n{DEMO_TEXT}")

    entities = predict(DEMO_TEXT)                                # распознавание
    print(f"\nНайдено сущностей: {len(entities)}")

    by_attr = defaultdict(list)                                  # группируем по атрибуту
    for attr, s, e in entities:
        by_attr[attr].append(DEMO_TEXT[s:e])

    print("\nПо атрибутам:")
    for attr in ATTRS:
        if attr in by_attr:
            vals = by_attr[attr]
            print(f"  {attr:<14} ({len(vals)}): {vals}")
        else:
            print(f"  {attr:<14} (0):   ⚠ не найдено")


# ========== ДЕМОНСТРАЦИЯ СКВОЗНОГО ЦИКЛА ==========
def demo_pipeline():
    """Сквозной цикл на демонстрационном тексте: распознать → обезличить."""
    from anonymize import recognize_and_anonymize
    from config import DEMO_TEXT

    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ: сквозной цикл (распознавание + обезличивание)")
    print("=" * 60)
    print(f"\nБыло:\n{DEMO_TEXT}\n")

    anonymized = recognize_and_anonymize(DEMO_TEXT)
    print(f"Стало:\n{anonymized}\n")

    # Второй вызов — проверка случайности (значения должны отличаться)
    anonymized2 = recognize_and_anonymize(DEMO_TEXT)
    print(f"Повторный вызов (проверка случайности):\n{anonymized2}")
    same = anonymized == anonymized2
    print(f"\n{'⚠ ВНИМАНИЕ: повторный вызов дал тот же результат' if same else '✅ Значения разные — случайность работает'}")


# ========== MAIN ==========
if __name__ == "__main__":
    # --- Загрузка данных (один раз) ---
    from dataset import load_factrueval
    print("Загрузка factRuEval...")
    examples, factrueval_test_docs = load_factrueval()
    print(f"   devset: {len(examples)} документов")
    print(f"   testset: {len(factrueval_test_docs)} документов")

    # --- Решение об обучении ---
    model_exists = os.path.exists(os.path.join(MODEL_OUT, "config.json"))
    if RETRAIN and model_exists:
        shutil.rmtree(MODEL_OUT, ignore_errors=True)         # удаляем старую модель
        print(f"🗑 Удалена старая модель: {MODEL_OUT}")
        model_exists = False

    if model_exists:
        print(f"ℹ Модель найдена в {MODEL_OUT} — обучение пропускается (RETRAIN=False)")
    else:
        run_training(examples)                           # обучаем
        pass
    
    # --- Оценка ---
    if not SKIP_TEST:
        run_evaluation(factrueval_test_docs)                 # метрики + assert'ы
    else:
        print("⏭ SKIP_TEST=True — оценка пропускается")

    # --- Демонстрация (всегда) ---
    demo_recognition()                                       # predict
    demo_pipeline()                                          # recognize_and_anonymize

    print("\n" + "=" * 60)
    print("✅ Готово.")
    print("=" * 60)