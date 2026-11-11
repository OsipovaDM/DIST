"""Глобальная конфигурация: пути, атрибуты, BIO-метки.

Никаких импортов из recognize/anonymize — иначе циклический импорт.
"""
import os

# ---- пути ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # c:\Dev\PDn
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_OUT = os.path.join(BASE_DIR, "models", "rubert_pii")

# ---- гиперпараметры ----
SEED = 42
MAX_LEN = 256

# ---- атрибуты чётного варианта ----
ATTRS = ["SURNAME", "NAME", "INN", "PASSPORT", "CREDIT_CARD",
         "OGRNIP", "IP", "EMAIL", "BIRTH_CERT", "KLADR"]

LABELS = ["O"] + [f"{p}-{a}" for a in ATTRS for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# ---- что умеет обезличивать ЛР2 ----
OBFUSCATABLE = {"SURNAME", "NAME", "EMAIL", "INN", "PHONE", "CREDIT_CARD"}