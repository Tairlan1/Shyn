"""Тесты style_verdict_text() - это функция, вокруг которой был весь
изначальный разговор: "AI-детектор дал 98%, а стилометрия - 68.3% на
H.G. Wells - это же противоречит друг другу". Эти тесты фиксируют
итоговое поведение, чтобы будущий рефакторинг случайно не вернул старый
баг (показ style% как значимого при явном ИИ-тексте)."""

from __future__ import annotations

from app import style_verdict_text, ai_verdict_text


def test_ai_red_gates_style_verdict_regardless_of_raw_percentage():
    """Даже при style_pct=95 (казалось бы, явное совпадение) и tier="green",
    если ai_tier="red" - вердикт должен быть "недостоверен", а не
    "совпадает". Это ТОЧНО тот сценарий из исходной жалобы пользователя."""
    text, stamp, cls = style_verdict_text(
        style_pct=68.3, tier="green", target_display="H.G. Wells",
        top_author_display="H.G. Wells", matches_top=True,
        ai_tier="red",
    )
    assert "НЕДОСТОВЕРЕН" in stamp
    assert cls == "bad"
    assert "неинформативно" in text


def test_novelty_unreliable_also_gates_verdict():
    text, stamp, cls = style_verdict_text(
        style_pct=90, tier="green", target_display="Mark Twain",
        top_author_display="Mark Twain", matches_top=True,
        ai_tier="green", novelty_pct=10.0, style_reliable=False,
    )
    assert "НЕДОСТОВЕРЕН" in stamp
    assert cls == "bad"


def test_normal_green_case_shows_match():
    text, stamp, cls = style_verdict_text(
        style_pct=90, tier="green", target_display="Jack London",
        top_author_display="Jack London", matches_top=True,
        ai_tier="green", style_reliable=True,
    )
    assert "СОВПАДАЕТ" in stamp
    assert cls == "ok"


def test_yellow_tier_mentions_partial_match():
    text, stamp, cls = style_verdict_text(
        style_pct=50, tier="yellow", target_display="Edgar Allan Poe",
        top_author_display="Mark Twain", matches_top=False,
        ai_tier="green", style_reliable=True,
    )
    assert "ЧАСТИЧНОЕ" in stamp
    assert cls == "warn"
    assert "Mark Twain" in text  # упоминает, к кому ближе на самом деле


def test_red_tier_without_ai_flag_is_mismatch_not_unreliable():
    """Стиль явно не совпал, но это НЕ ИИ и не аномалия - должно быть
    спокойное "не совпадает", а не тревожное "недостоверен"."""
    text, stamp, cls = style_verdict_text(
        style_pct=10, tier="red", target_display="Arthur Conan Doyle",
        top_author_display="Mark Twain", matches_top=False,
        ai_tier="green", style_reliable=True,
    )
    assert stamp == "СТИЛЬ НЕ<br>СОВПАДАЕТ"
    assert cls == "bad"
    assert "НЕДОСТОВЕРЕН" not in stamp


def test_ai_verdict_red_is_never_worded_as_proof():
    """AI Detection никогда не должен формулироваться как доказанный
    факт - только как повод для проверки (см. справка для комиссии)."""
    text, stamp, cls = ai_verdict_text(95, "red")
    assert cls == "bad"
    assert "доказательство" not in text.lower() or "не окончательное доказательство" in text
    assert "повод" in text.lower()


def test_ai_verdict_green_still_notes_limitations():
    text, stamp, cls = ai_verdict_text(2, "green")
    assert cls == "ok"
    assert "ограничения" in text.lower()

