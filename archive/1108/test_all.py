"""Юнит-тесты для пакета pdn.

Запуск: pytest tests/ -v
"""
import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OBFUSCATABLE
from recognize import predict, validate_inn, validate_luhn
from anonymize import (
    anon, recognize_and_anonymize,
    obfuscate_email, obfuscate_inn, obfuscate_phone, obfuscate_credit_card,
    generate_name_with_morphology, generate_surname_with_morphology,
    get_grammemes,
)


def test_validators():
    """Контрольные разряды: эталонные значения проходят, мусор — нет."""
    assert validate_inn("770698959438")
    assert validate_luhn("4561261212345467")
    assert not validate_inn("77069895943")


def test_email():
    """Домен email сохраняется, локальная часть меняется."""
    for _ in range(20):
        src = "user.name@corp.example.ru"
        out = obfuscate_email(src)
        assert "@corp.example.ru" in out
        assert out != src


def test_inn():
    """Префикс ИНН сохранён, контроль валиден."""
    for _ in range(20):
        out = obfuscate_inn("770698959438")
        d = re.sub(r"\D", "", out)
        assert d[:4] == "7706"
        assert validate_inn(d)
        assert d != "770698959438"


def test_phone():
    """4 первые цифры и нецифровые символы сохранены."""
    for _ in range(20):
        src = "+7 (999) 777-12-34"
        out = obfuscate_phone(src)
        assert re.findall(r"\d", out)[:4] == re.findall(r"\d", src)[:4]
        assert [c for c in out if not c.isdigit()] == [c for c in src if not c.isdigit()]
        assert out != src


def test_credit_card():
    """BIN и суффикс сохранены, Луна валиден."""
    for _ in range(20):
        src = "4561 2612 1234 5467"
        out = obfuscate_credit_card(src)
        d = re.sub(r"\D", "", out)
        assert d[:6] == "456126"
        assert d[-4:] == "5467"
        assert validate_luhn(d)
        assert d != "4561261212345467"


def test_morphology():
    """Род и падеж сохраняются (см. ЛР2: проблема с неоднозначными именами)."""
    cases = [
        ("Иван",     "masc", "nomn", generate_name_with_morphology),
        ("Ивана",    "masc", "gent", generate_name_with_morphology),
        ("Марию",    "femn", "accs", generate_name_with_morphology),
        ("Петровой", "femn", "gent", generate_surname_with_morphology),
    ]
    for word, gender, case, fn in cases:
        for _ in range(10):
            new = fn(word)
            gr = get_grammemes(new)
            assert gr["gender"] == gender, f"{word} → {new}"
            assert gr["case"] == case, f"{word} → {new}"


def test_randomness():
    """50 вызовов — больше одного уникального результата."""
    results = {anon("Иван", [("NAME", 0, 4)]) for _ in range(50)}
    assert len(results) > 1


def test_anon_edges():
    """Пустой текст, мусор, некорректные индексы — не падает."""
    assert anon("", []) == ""
    assert anon("abc", [("NAME", 5, 10)]) == "abc"
    assert anon("abc", [("NAME", -1, 2)]) == "abc"
    assert anon("abc", [("UNKNOWN", 0, 3)]) == "abc"


def test_predict():
    """predict находит SURNAME, NAME, EMAIL, INN."""
    res = predict("Иванов Иван, ivan@mail.ru, ИНН 770698959438.")
    attrs = {e[0] for e in res}
    assert "SURNAME" in attrs
    assert "NAME" in attrs
    assert "EMAIL" in attrs
    assert "INN" in attrs


def test_recognize_and_anonymize():
    """Сквозной цикл: ФИО и email заменяются, паспорт не в OBFUSCATABLE."""
    raw = "Иванов Иван, ivan@mail.ru, паспорт 45 06 123456"
    out = recognize_and_anonymize(raw)
    assert "Иванов" not in out and "Иван" not in out
    assert "ivan@mail.ru" not in out
    # Паспорт не меняется — его нет в OBFUSCATABLE
    assert "45 06 123456" in out