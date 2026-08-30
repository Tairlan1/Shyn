"""Тесты границ цветовых порогов. Эти два числа (33/80 для стиля,
25/90 для AI) были предметом долгого обсуждения и намеренно
асимметричны (см. комментарии в app.py) - тесты фиксируют это поведение,
чтобы случайная правка констант не осталась незамеченной."""

from __future__ import annotations

import app


def test_style_tier_boundaries():
    assert app.tier_of(0) == "red"
    assert app.tier_of(32.9) == "red"
    assert app.tier_of(33.0) == "red"  # граница включена в красный (<=), не в жёлтый
    assert app.tier_of(33.1) == "yellow"
    assert app.tier_of(60) == "yellow"
    assert app.tier_of(79.9) == "yellow"
    assert app.tier_of(80.0) == "green"
    assert app.tier_of(100) == "green"


def test_ai_tier_boundaries_and_inverted_semantics():
    """Для AI% семантика цвета ОБРАТНАЯ: низкий % - хорошо (зелёный),
    высокий - тревожно (красный). Пороги здесь СВОИ, отдельные от
    порогов стиля (см. AI_BAND_RED_MAX/GREEN_MIN) - AI-порог красного
    заметно консервативнее (90, а не 80), т.к. цена ложного обвинения
    в ИИ выше, чем у заниженного style%."""
    assert app._ai_tier(0) == "green"
    assert app._ai_tier(25.0) == "green"
    assert app._ai_tier(50) == "yellow"
    assert app._ai_tier(89.9) == "yellow"
    assert app._ai_tier(90.0) == "red"
    assert app._ai_tier(100) == "red"


def test_ai_thresholds_are_more_conservative_than_style():
    """Явно фиксирует намеренную асимметрию - если кто-то случайно
    уравняет эти пороги при рефакторинге, тест должен упасть."""
    assert app.AI_BAND_GREEN_MIN > app.BAND_GREEN_MIN
    assert app.AI_BAND_RED_MAX < app.BAND_RED_MAX
