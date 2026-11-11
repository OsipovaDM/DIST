"""Глобальная конфигурация проекта: параметры, справочники, пути"""
import os
import re

# ========== ПУТИ ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))         # корень проекта
DATA_DIR = os.path.join(BASE_DIR, "data")                     # куда качается factRuEval
MODEL_OUT = os.path.join(BASE_DIR, "models", "rubert_pii")    # куда сохраняется BERT

# ========== ИСТОЧНИКИ ДАННЫХ ==========
FACT_RUEVAL_URL = "https://github.com/dialogue-evaluation/factRuEval-2016/archive/master.zip"  # архив factRuEval

# ========== РЕЖИМЫ РАБОТЫ main.py ==========
RETRAIN = False         # True → удалить модель и обучить заново
SKIP_TEST = True       # True → не считать метрики

# ========== СХЕМА ДАННЫХ (обучение модели) ==========
ATTRS = ["SURNAME", "NAME", "INN", "PASSPORT", "CREDIT_CARD",
         "OGRNIP", "IP", "EMAIL", "BIRTH_CERT", "KLADR"]       # 10 атрибутов чётного варианта
BERT_ATTRS = ["SURNAME", "NAME"]
BERT_LABELS = ["O"] + [f"{p}-{a}" for a in BERT_ATTRS for p in ("B", "I")]
BERT_LABEL2ID = {l: i for i, l in enumerate(BERT_LABELS)}
BERT_ID2LABEL = {i: l for l, i in BERT_LABEL2ID.items()}
MAX_LEN = 512                                                  # длина последовательности токенов
TYPE_TO_ATTR = {"surname": "SURNAME", "name": "NAME"}          # маппинг типов из corus

# ========== СПРАВОЧНИКИ ==========
PASSPORT_REGION_WHITELIST = {
    1,3,4,5,7,8,10,11,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,
    32,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
    54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,
    75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,
    96,97,98,99,
}                                                              # коды регионов РФ (PASSPORT/KLADR)
INN_COEFF_C11 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]               # коэффициенты 11-й цифры ИНН
INN_COEFF_C12 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]            # коэффициенты 12-й цифры ИНН

# ========== ГИПЕРПАРАМЕТРЫ ОБУЧЕНИЯ (обучение модели) ==========
SEED = 42                                # воспроизводимость random/numpy/torch
BASE_MODEL = "DeepPavlov/rubert-base-cased"  # предобученный чекпоинт для дообучения
NUM_EPOCHS = 10                          # число полных проходов по train
BATCH_SIZE = 16                          # размер батча на устройство
LR = 2e-5                                # скорость обучения оптимизатора
WEIGHT_DECAY = 0.01                      # L2-регуляризация
WARMUP_RATIO = 0.1                       # доля шагов на разогрев LR
LR_SCHEDULER_TYPE = "cosine"             # тип затухания LR
LOGGING_STEPS = 50                       # как часто печатать loss
SAVE_TOTAL_LIMIT = 1                     # сколько чекпоинтов хранить
DATALOADER_NUM_WORKERS = 0               # число воркеров DataLoader (0 = основной поток)
REPORT_TO = "none"                       # куда слать логи (none = никуда)
METRIC_FOR_BEST_MODEL = "f1"             # по какой метрике выбирать лучшую эпоху
VALID_SIZE = 0.2                         # доля valid от train+valid
TARGET_F1 = 0.70                         # None → не останавливать; число → стоп, когда eval_f1 ≥ этого

# ========== ПАРАМЕТРЫ ИНФЕРЕНСА (без переобучения) ==========
BERT_DEVICE = -1                          # -1 = CPU, 0 = первая GPU
AGGREGATION_STRATEGY = "simple"           # как склеивать BIO-теги в единый спан
CHUNK_STRIDE_RATIO = 0.5                  # перекрытие чанков при длинном тексте

# ========== ПАРАМЕТРЫ ОБЕЗЛИЧИВАНИЯ ==========
OBFUSCATABLE = {"SURNAME", "NAME", "EMAIL", "INN", "PHONE", "CREDIT_CARD"}  # какие атрибуты заменяем
EMAIL_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"  # алфавит для локальной части email
EMAIL_MIN_LOCAL_LEN = 4                                  # мин. длина новой локальной части
MAX_NAME_GEN_ATTEMPTS = 200                              # попыток подбора имени с морфологией

# ========== ТЕСТИРОВАНИЕ ==========
N_PER_ATTR = 60               # текстов на каждый атрибут
THRESHOLD_WEIGHTED_F1 = 0.85  # порог weighted F1
THRESHOLD_MEAN_F1 = 0.80      # порог mean F1
MIN_TEST_DOCS = 1             # минимум документов в factRuEval-тесте

# ========== УТИЛИТЫ ==========
def digits_only(s: str) -> str:
    """Оставляет в строке только цифры — для проверки и генерации номеров."""
    return re.sub(r"\D", "", s)