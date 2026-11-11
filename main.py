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
    """Показывает распознавание на тех же шаблонах, что в тестировании.

    Генерирует по DEMO_SAMPLES_PER_ATTR примеров на каждый атрибут через
    ту же _build_test_texts(), что и evaluate(). При DEMO_SAMPLES_PER_ATTR=3
    покажет все форматы разделителей (compact/space/dash) и все шаблоны.
    """
    from recognize import predict
    from evaluation import _build_test_texts
    from config import DEMO_SAMPLES_PER_ATTR

    print("\n" + "=" * 60)
    print(f"ДЕМОНСТРАЦИЯ: распознавание ПДн (по {DEMO_SAMPLES_PER_ATTR} примеру на атрибут)")
    print("=" * 60)

    test_texts = _build_test_texts(DEMO_SAMPLES_PER_ATTR)   # те же генераторы, что в evaluate

    # Группируем по атрибуту — чтобы вывод был по разделам
    by_attr = {}
    for text, attr, value in test_texts:
        by_attr.setdefault(attr, []).append((text, value))

    for attr, samples in by_attr.items():
        print(f"\n--- {attr} ---")
        for text, expected in samples:
            entities = predict(text)
            # Проверяем, нашёлся ли ожидаемый атрибут
            found = any(e_attr == attr for e_attr, _, _ in entities)
            mark = "✅" if found else "❌"
            print(f"{mark} {text}")
            print(f"   ожидалось: {attr}='{expected}'")
            if entities:
                for e_attr, s, e in entities:
                    print(f"   найдено:   {e_attr}='{text[s:e]}' [{s}:{e}]")
            else:
                print("   найдено:   (ничего)")


# ========== ДЕМОНСТРАЦИЯ ОБЕЗЛИЧИВАНИЯ ==========
def demo_anonymization():
    """Показывает синонимическую замену по готовому списку сущностей."""
    from anonymize import anon                               # импорт здесь — не тянет pymorphy при import main
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ: обезличивание по готовому списку сущностей")
    print("=" * 60)

    samples = [
        ("Иван получил кредит.", [("NAME", 0, 4)]),          # простое имя
        ("Иванов Иван, ivan@mail.ru.", [("SURNAME", 0, 6), ("NAME", 7, 11), ("EMAIL", 13, 25)]),
        ("ИНН 770698959438, карта 4561 2612 1234 5467.",
         [("INN", 4, 16), ("CREDIT_CARD", 24, 43)]),
        ("Позвоните Петровой Марии.", [("SURNAME", 10, 18), ("NAME", 19, 24)]),  # падежи
    ]
    for text, entities in samples:
        anonymized = anon(text, entities)                    # замена
        print(f"\nБыло:  {text}")
        print(f"Стало: {anonymized}")
        # Второй вызов — проверка случайности
        anonymized2 = anon(text, entities)
        print(f"Ещё:   {anonymized2}  {'(≠ первого)' if anonymized2 != anonymized else '⚠ совпало'}")


# ========== ДЕМОНСТРАЦИЯ СКВОЗНОГО ЦИКЛА ==========
def demo_pipeline():
    """Показывает полный цикл: распознать → отфильтровать → обезличить."""
    from anonymize import recognize_and_anonymize            # импорт здесь — ленивая загрузка BERT
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ: сквозной цикл (распознавание + обезличивание)")
    print("=" * 60)

    samples = [
        "Иванов Иван, почта ivan.petrov@example.ru.",
        "ИНН 770698959438, карта 4561 2612 1234 5467.",
        # Паспорт и ОГРНИП НЕ в OBFUSCATABLE — должны остаться
        "Паспорт 45 06 123456, ОГРНИП 304500116000157.",
        "Петрова Мария, petrova@corp.ru, ИНН 500100732259.",
    ]
    for text in samples:
        print(f"\nБыло:  {text}")
        print(f"Стало: {recognize_and_anonymize(text)}")


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
    demo_anonymization()                                     # anon
    demo_pipeline()                                          # recognize_and_anonymize

    print("\n" + "=" * 60)
    print("✅ Готово.")
    print("=" * 60)