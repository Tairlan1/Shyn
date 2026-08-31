#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis.py
=====================================================================
Основная аналитическая логика платформы: цветовые пороги, разбиение
текста на окна для подсветки, честная оценка достоверности по длине
текста, сам анализ (analyze_text) и формулировки вердиктов.

Вынесено из app.py при переходе на модульную архитектуру - app.py
теперь отвечает только за Flask-роуты и запуск, вся содержательная
логика анализа живёт здесь и не зависит от Flask вообще (что делает её
тривиально тестируемой и потенциально переиспользуемой, например,
если понадобится третий "вход" к этой же логике помимо app.py/
api_analyze.py).

ВАЖНО: используем `import config` и обращаемся как `config.PIPELINE`,
`config.MODEL_DIR` и т.д., а не `from config import PIPELINE` - см.
подробное объяснение в докстринге config.py. Функции здесь читают эти
значения В МОМЕНТ ВЫЗОВА, поэтому корректно видят то, что app.main()
устанавливает при старте (или что тесты подставляют через config.X = ...).
"""

from __future__ import annotations

import html
import json
import re

import numpy as np

import ai_detector
import ai_heuristics
import config
import train_model
from train_model import predict_author


def tier_of(pct: float) -> str:
    if pct <= config.BAND_RED_MAX:
        return "red"
    if pct >= config.BAND_GREEN_MIN:
        return "green"
    return "yellow"


def _ai_tier(ai_pct: float) -> str:
    """Для AI% семантика цвета обратная относительно 'соответствия стилю':
    низкий AI% - хорошо (зелёный/оригинальный текст), высокий - тревожно
    (красный/вероятно ИИ). Пороги здесь СВОИ (AI_BAND_*), отдельные от
    порогов совпадения стиля - у ложного обвинения в ИИ намного выше цена
    ошибки, чем у заниженного Shyndyq %, поэтому по умолчанию AI-порог
    красного заметно консервативнее (см. config.AI_BAND_RED_MAX/AI_BAND_GREEN_MIN)."""
    if ai_pct >= config.AI_BAND_GREEN_MIN:
        return "red"
    if ai_pct <= config.AI_BAND_RED_MAX:
        return "green"
    return "yellow"


# =============================================================================
# Разбиение исходного (несжатого/неочищенного) текста на окна для подсветки,
# с сохранением точных смещений символов в ИСХОДНОМ тексте пользователя.
# =============================================================================

def _split_paragraph_blocks(raw_text: str) -> list[tuple[int, int, str]]:
    """Возвращает [(start, end, text)] блоков-абзацев, разделённых пустой
    строкой, с точными смещениями в raw_text (никакой очистки/изменения
    текста - подсветка должна ложиться на исходные символы)."""
    blocks = []
    pos = 0
    for part in re.split(r"(\n\s*\n)", raw_text):
        if part == "":
            continue
        if re.fullmatch(r"\n\s*\n", part):
            pos += len(part)
            continue
        start = pos
        end = pos + len(part)
        if part.strip():
            blocks.append((start, end, part))
        pos = end
    return blocks


def build_windows(raw_text: str) -> list[dict]:
    """Группирует абзацы в окна ~WINDOW_TARGET_WORDS слов, сохраняя точные
    смещения в исходном тексте для последующей подсветки."""
    blocks = _split_paragraph_blocks(raw_text)
    windows = []
    cur_start = None
    cur_end = None
    cur_words = 0

    def flush():
        nonlocal cur_start, cur_end, cur_words
        if cur_start is not None:
            windows.append({
                "start": cur_start,
                "end": cur_end,
                "text": raw_text[cur_start:cur_end],
                "word_count": cur_words,
            })
        cur_start, cur_end, cur_words = None, None, 0

    for start, end, text in blocks:
        wc = len(text.split())
        if cur_start is None:
            cur_start, cur_end, cur_words = start, end, wc
        else:
            cur_end = end
            cur_words += wc
        if cur_words >= config.WINDOW_TARGET_WORDS:
            flush()
    flush()

    # Последнее окно может быть слишком маленьким - сливаем с предыдущим.
    if len(windows) >= 2 and windows[-1]["word_count"] < config.WINDOW_MIN_WORDS:
        last = windows.pop()
        prev = windows[-1]
        prev["end"] = last["end"]
        prev["text"] = raw_text[prev["start"]:prev["end"]]
        prev["word_count"] += last["word_count"]

    return windows


# =============================================================================
# Подсветка: один проход по тексту, каждый фрагмент оборачивается в <span>,
# несущий ОБА независимых тега (data-style / data-ai). Какой из них видно -
# решает чистый CSS/JS-переключатель на странице, без повторной отрисовки
# текста и без прыжков скролла.
# =============================================================================

def render_dual_highlight_html(raw_text: str, windows: list[dict]) -> str:
    parts = []
    last_pos = 0
    for w in windows:
        if w["start"] > last_pos:
            parts.append(html.escape(raw_text[last_pos:w["start"]]))
        segment = html.escape(w["text"])
        style_pct = round(w["style_score"] * 100, 1)
        ai_pct = round(w["ai_score"] * 100, 1)
        parts.append(
            f'<span class="frag" data-style="{tier_of(style_pct)}" '
            f'data-style-pct="{style_pct}" data-ai="{_ai_tier(ai_pct)}" '
            f'data-ai-pct="{ai_pct}">{segment}</span>'
        )
        last_pos = w["end"]
    if last_pos < len(raw_text):
        parts.append(html.escape(raw_text[last_pos:]))
    return "".join(parts)


# =============================================================================
# Достоверность оценки в зависимости от длины текста - для непрофессионалов.
# Использует ЧЕСТНЫЕ (held-out, на невиданных книгах) метрики по масштабам
# фрагмента из model_meta.json multiscale-модели (см. train_model.py
# build-multiscale и compare_models.py). Если активная модель обучена по
# старой схеме (только ~1500 слов, без held_out_metrics_by_scale) -
# используется общий, менее точный, но честно консервативный текст.
# =============================================================================

# (нижняя_граница_слов, ключ_масштаба_в_model_meta.json, понятная подпись)
_WORD_COUNT_BANDS = [
    (900, "large", "полноценная работа (900+ слов)"),
    (600, "medium", "объёмный текст (600-900 слов)"),
    (250, "semi_small", "средний по объёму текст (250-600 слов)"),
    (60, "window", "короткий текст (60-250 слов)"),
    (0, "phrase", "очень короткий текст/фраза (менее 60 слов)"),
]

_MODEL_META_CACHE: dict | None = None


def _load_model_meta() -> dict:
    global _MODEL_META_CACHE
    if _MODEL_META_CACHE is None:
        meta_path = config.MODEL_DIR / "model_meta.json"
        try:
            _MODEL_META_CACHE = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _MODEL_META_CACHE = {}
    return _MODEL_META_CACHE


def confidence_for_word_count(n_words: int) -> dict:
    """Возвращает понятное (без жаргона) описание того, насколько можно
    доверять оценке стиля ИМЕННО для текста такой длины - на основе честных
    held-out метрик по масштабам (не общей точности модели "в среднем"),
    если они есть у текущей модели."""
    meta = _load_model_meta()
    by_scale = meta.get("held_out_metrics_by_scale")

    band_label = next(label for lo, _, label in _WORD_COUNT_BANDS if n_words >= lo)
    scale_key = next(key for lo, key, _ in _WORD_COUNT_BANDS if n_words >= lo)

    if not by_scale or scale_key not in by_scale:
        # Модель без по-масштабной валидации (старая, только 1500 слов) -
        # честно говорим, что для текста короче ~1000 слов достоверность
        # оценки нам неизвестна, вместо того чтобы придумывать число.
        if n_words >= 900:
            return {"band_label": band_label, "level": "known",
                    "accuracy_pct": round(meta.get("held_out_accuracy", 0) * 100, 1),
                    "text": "Модель проверялась именно на текстах такой длины - "
                            "оценке для такого объёма можно доверять."}
        return {"band_label": band_label, "level": "unknown", "accuracy_pct": None,
                "text": "Эта модель обучена и проверена только на объёмных текстах "
                        "(около 1500 слов) - для более коротких текстов её "
                        "достоверность отдельно не измерялась, относитесь к "
                        "результату с осторожностью."}

    acc = round(by_scale[scale_key]["accuracy"] * 100, 1)
    if acc >= 90:
        level, text = "high", "Модель проверялась на текстах такой длины и почти всегда угадывает автора верно."
    elif acc >= 70:
        level, text = "good", "Модель проверялась на текстах такой длины и в большинстве случаев угадывает автора верно."
    elif acc >= 50:
        level, text = "moderate", "Для текста такой длины модель ошибается заметно чаще - относитесь к результату как к ориентиру, а не к доказательству."
    else:
        level, text = "low", "Для текста такой длины модель часто ошибается - на этот результат не стоит полагаться серьёзно, нужен более длинный текст для надёжной оценки."
    return {"band_label": band_label, "level": level, "accuracy_pct": acc, "text": text}


# =============================================================================
# Основной анализ (два НЕЗАВИСИМЫХ расчёта: стиль через ML-пайплайн,
# ИИ-индикатор через отдельную эвристику - друг на друга не влияют).
# =============================================================================

def analyze_text(raw_text: str, target_author: str) -> dict:
    if not raw_text or not raw_text.strip():
        raise ValueError("Текст пуст или не удалось извлечь текст из файла.")

    author_classes = list(config.LABEL_ENCODER.classes_)
    if target_author not in author_classes:
        raise ValueError(f"Неизвестный автор: {target_author}")
    target_idx = author_classes.index(target_author)

    # ---- 1) Официальный документ-уровневый вердикт (как в CLI predict) ----
    doc_result = predict_author(raw_text, model_dir=config.MODEL_DIR)
    style_overall_pct = 0.0
    for r in doc_result["ranking"]:
        if r["author"] == target_author:
            style_overall_pct = round(r["probability"] * 100, 1)
            break

    # ---- 2) Окна для подсветки (более мелкая гранулярность, приближённо) ----
    windows = build_windows(raw_text)
    if not windows:
        raise ValueError("Не удалось выделить фрагменты текста для анализа.")

    window_texts = [w["text"] for w in windows]

    # --- независимый расчёт №1: авторский стиль (ML-пайплайн) ---
    proba = config.PIPELINE.predict_proba(window_texts)  # (n_windows, n_authors)
    for i, w in enumerate(windows):
        w["style_score"] = float(proba[i][target_idx])

    # --- независимый расчёт №1b: детектор "вне распределения" (novelty) ---
    # Отвечает не "какой автор ближе", а "похож ли текст вообще на корпус,
    # на котором обучалась стилевая модель" - см. train_model.py. Если
    # OOD_DETECTOR отсутствует (старая модель без ood_novelty.joblib),
    # window_novelty будет None и весь дальнейший код должен это учитывать.
    window_novelty = None
    if config.OOD_DETECTOR is not None:
        window_novelty = train_model.novelty_pct_scores(window_texts, config.PIPELINE, config.OOD_DETECTOR)
        for i, w in enumerate(windows):
            w["novelty_pct"] = round(float(window_novelty[i]), 1)

    # --- независимый расчёт №2: обученный классификатор "человек/ИИ",
    #     с откатом на эвристику для слишком коротких фрагментов или если
    #     модель не обучена (см. train_ai_detector.py) ---
    ai_sources_used = set()
    for w in windows:
        res = ai_detector.score_fragment(w["text"], config.PROJECT_DIR)
        if res is not None:
            w["ai_score"] = res["ai_score"]
            w["ai_markers"] = ai_heuristics.score_fragment(w["text"])["marker_hits"]
            ai_sources_used.add("trained_classifier")
        else:
            res = ai_heuristics.score_fragment(w["text"])
            w["ai_score"] = res["ai_score"]
            w["ai_markers"] = res["marker_hits"]
            ai_sources_used.add("heuristic_fallback")

    ai_overall = float(np.mean([w["ai_score"] for w in windows]))
    ai_source_label = (
        "обученная модель" if ai_sources_used == {"trained_classifier"}
        else "эвристика (резерв)" if ai_sources_used == {"heuristic_fallback"}
        else "обученная модель + эвристика для коротких фрагментов"
    )

    doc_html = render_dual_highlight_html(raw_text, windows)

    novelty_pct = doc_result.get("novelty_pct")  # None, если модель без OOD-детектора
    # ВАЖНО: novelty_pct пока НЕ используется для автоматического понижения
    # вердикта (см. NOVELTY_GATING_ENABLED ниже) - эмпирическая проверка на
    # реальных текстах показала, что density-based детектор в текущей
    # реализации может ошибаться в опасную сторону: подлинный, но стилево
    # яркий текст человека получает БОЛЕЕ высокий сигнал "аномальности", чем
    # сглаженный ИИ-текст, который как раз статистически ближе к "среднему"
    # по корпусу. Автоматически понижать вердикт студента по такому сигналу
    # без валидации на полном корпусе - неприемлемый риск. Значение всё
    # равно считается и показывается (для информации/ручной проверки), но
    # ЕДИНСТВЕННЫЙ сигнал, который сейчас автоматически шлюзует вердикт -
    # это обученный AI-детектор (ai_tier), который такую проверку прошёл.
    style_reliable = (not config.NOVELTY_GATING_ENABLED) or novelty_pct is None \
        or novelty_pct >= config.NOVELTY_UNRELIABLE_MAX

    return {
        "target_author": target_author,
        "target_author_display": config.AUTHOR_DISPLAY_NAMES.get(target_author, target_author),
        "style_overall_pct": style_overall_pct,
        "style_tier": tier_of(style_overall_pct),
        "style_predicted_author": doc_result["predicted_author"],
        "style_predicted_author_display": config.AUTHOR_DISPLAY_NAMES.get(
            doc_result["predicted_author"], doc_result["predicted_author"]),
        "style_ranking": [
            {"author": config.AUTHOR_DISPLAY_NAMES.get(r["author"], r["author"]),
             "pct": round(r["probability"] * 100, 1)}
            for r in doc_result["ranking"]
        ],
        "novelty_pct": novelty_pct,
        "style_reliable": style_reliable,
        "ai_overall_pct": round(ai_overall * 100, 1),
        "ai_tier": _ai_tier(round(ai_overall * 100, 1)),
        "ai_source_label": ai_source_label,
        "doc_html": doc_html,
        "n_windows": len(windows),
        "n_words": len(raw_text.split()),
        "confidence": confidence_for_word_count(len(raw_text.split())),
    }


# =============================================================================
# Вердикты (детерминированные, на основе тех же порогов 33/80).
# =============================================================================

def style_verdict_text(style_pct: float, tier: str, target_display: str,
                        top_author_display: str, matches_top: bool,
                        ai_tier: str = "green", novelty_pct: float | None = None,
                        style_reliable: bool = True) -> tuple[str, str, str]:
    """Вердикт по авторскому стилю - но теперь НЕ полностью независимый от
    двух других сигналов (AI% и novelty%). Раньше высокий style_pct мог
    показываться как "совпадение" даже для явно ИИ-сгенерированного или
    статистически аномального текста - см. историю проблемы. Правило
    приоритета: если ai_tier=="red" (детектор ИИ уверен) ИЛИ style_reliable
    == False (текст вне распределения обучающего корпуса - см.
    train_model.novelty_pct_scores), то raw style_pct не показывается как
    значимое совпадение, каким бы высоким он ни был - потому что вычислен
    на признаках, которые в этом случае ничего не доказывают."""
    if ai_tier == "red":
        text = (f"Текст с высокой вероятностью сгенерирован ИИ (см. вкладку "
                f"AI Detection). Совпадение стиля с {target_display} в этом "
                f"случае неинформативно и не может расцениваться как "
                f"подтверждение авторства, даже если процент совпадения выше.")
        return text, "СТИЛЬ<br>НЕДОСТОВЕРЕН", "bad"
    if not style_reliable:
        text = (f"Текст статистически нетипичен для корпуса, на котором "
                f"обучалась модель (novelty {novelty_pct:.0f}%) - он не похож "
                f"ни на одного из пяти эталонных авторов настолько, чтобы "
                f"атрибуции можно было доверять. Сравнение с {target_display} "
                f"в данном случае недостоверно; рекомендуется ручная проверка.")
        return text, "СТИЛЬ<br>НЕДОСТОВЕРЕН", "bad"
    if tier == "green":
        text = (f"Стиль текста с высокой уверенностью совпадает с работами "
                f"{target_display} — модель уверенно относит документ к этому автору "
                f"по большинству проанализированных фрагментов.")
        return text, "СТИЛЬ<br>СОВПАДАЕТ", "ok"
    if tier == "yellow":
        extra = "" if matches_top else f" Ближе всего по модели — {top_author_display}."
        text = (f"Стиль текста частично совпадает с работами {target_display}: часть "
                f"фрагментов стилистически близка, часть — заметно отличается.{extra}")
        return text, "ЧАСТИЧНОЕ<br>СОВПАДЕНИЕ", "warn"
    text = (f"Стиль текста существенно отличается от стиля {target_display}. "
            f"Модель относит документ скорее к {top_author_display}.")
    return text, "СТИЛЬ НЕ<br>СОВПАДАЕТ", "bad"


def ai_verdict_text(ai_pct: float, tier: str) -> tuple[str, str, str]:
    if tier == "red":
        text = ("Эвристика отмечает признаки, характерные для текста, "
                "сгенерированного ИИ, на значительной части работы. Это повод для "
                "ручной проверки, а не окончательное доказательство.")
        return text, "ТРЕБУЕТ<br>ПРОВЕРКИ", "bad"
    if tier == "yellow":
        text = ("В отдельных фрагментах обнаружены отдельные признаки, "
                "характерные для ИИ-генерации (лексика, ровный ритм предложений). "
                "Однозначного вывода эвристика не даёт.")
        return text, "ЕСТЬ<br>ЗАМЕЧАНИЯ", "warn"
    text = ("Существенных признаков ИИ-генерации не обнаружено (по оценке "
            "обученного классификатора). Учтите ограничения модели — см. "
            "пояснение на странице разбора.")
    return text, "БЕЗ<br>ЗАМЕЧАНИЙ", "ok"
