#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — Shyndyq
=====================================================================
Демонстрационная университетская платформа поверх train_model.py:

  - студент выбирает предмет (каждый предмет привязан к одному из 5
    эталонных авторских стилей) и загружает работу (.docx / .pdf / .txt);
  - анализ идёт в фоновом потоке (job), пока фронтенд показывает
    анимацию стадий "Shyndyq анализирует работу...";
  - в таблице "Мои работы" рядом показаны два кликабельных процента:
    Shyndyq % (соответствие авторскому стилю) и AI % (эвристический
    индикатор ИИ-генерации), оба цветокодированы (красный/жёлтый/зелёный);
  - страница разбора показывает полный текст работы один раз и
    переключается между подсветкой "Авторский стиль" / "AI Detection"
    БЕЗ перезагрузки текста - переключаются только CSS-классы поверх
    уже отрисованных фрагментов, так что скролл и раскладка не прыгают.
  - оба вида анализа считаются полностью независимо друг от друга.

Запуск:

    python app.py --model-dir ./model

Затем открыть http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import threading
import uuid
from datetime import date
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, redirect, render_template, request, url_for

import ai_detector
import ai_heuristics
import doc_extract
import storage
import train_model
from train_model import load_model, predict_author

# Модель была обучена запуском `python train_model.py ...` напрямую, поэтому
# joblib сохранил классы/функции, использованные в pipeline, с привязкой к
# модулю `__main__` (а не `train_model`). Чтобы joblib.load смог их найти
# при запуске уже из app.py, публикуем те же объекты в sys.modules['__main__'].
import __main__ as _main_module
for _name in ("StylometricFeaturizer", "DenseTransformer",
              "extract_stylometric_features", "FUNCTION_WORDS"):
    if hasattr(train_model, _name):
        setattr(_main_module, _name, getattr(train_model, _name))

app = Flask(__name__)

PIPELINE = None
LABEL_ENCODER = None
OOD_DETECTOR = None  # None = модель без novelty-детектора (старые артефакты)
MODEL_DIR = Path("./model")
PROJECT_DIR = Path(__file__).parent

# Пороги цветовой классификации (в процентах) - раздельно для стиля и AI
# Detection, т.к. у них РАЗНАЯ цена ошибки: заниженный Shyndyq % просто
# менее интересен, а ложный красный AI-вердикт на честной работе - серьёзное
# обвинение. Поэтому AI-порог красного заметно консервативнее (90, а не 80).
BAND_RED_MAX = 33.0
BAND_GREEN_MIN = 80.0

AI_BAND_RED_MAX = 25.0
AI_BAND_GREEN_MIN = 90.0

# Размер окна для подсветки фрагментов (компромисс между точностью модели,
# обученной на ~1500-словных окнах, и желаемой гранулярностью подсветки).
WINDOW_TARGET_WORDS = 180
WINDOW_MIN_WORDS = 60

# Ниже этого novelty% стилевая атрибуция считалась бы недостоверной вне
# зависимости от % совпадения с целевым автором - см. train_model.novelty_pct_scores().
NOVELTY_UNRELIABLE_MAX = 35.0

# ВЫКЛЮЧЕНО ПО УМОЛЧАНИЮ. Density-based novelty-детектор эмпирически
# проверен на небольшой выборке реальных данных проекта и показал, что
# может занижать novelty% для подлинного, стилево яркого текста человека
# сильнее, чем для сглаженного ИИ-текста - т.е. способен работать в опасную
# для честного студента сторону. Включайте только после того, как
# validate_novelty_detector.py (см. отдельный скрипт) на ПОЛНОМ корпусе
# и реальных примерах ИИ-текста покажет, что человеческий held-out текст
# стабильно получает более высокий novelty%, чем ИИ-текст.
NOVELTY_GATING_ENABLED = False

AUTHOR_DISPLAY_NAMES = {
    "ArthurConanDoyle": "Arthur Conan Doyle",
    "EdgarAllanPoe": "Edgar Allan Poe",
    "H.G.Wells": "H.G. Wells",
    "JackLondon": "Jack London",
    "MarkTwain": "Mark Twain",
}

# Каждый "предмет" на платформе привязан к одному эталонному авторскому
# стилю - так учебный сценарий (выбор предмета) естественно ложится на
# единственную задачу, которую умеет решать модель (сравнение с одним
# из 5 обученных стилей).
SUBJECTS = [
    {"code": "LIT-201", "author": "ArthurConanDoyle",
     "name": "Детективная проза: стиль А. К. Дойла",
     "blurb": "Мастерская детективного рассказа на примере Шерлока Холмса."},
    {"code": "LIT-202", "author": "EdgarAllanPoe",
     "name": "Готическая новелла: стиль Э. А. По",
     "blurb": "Атмосфера, ужас и сжатая форма новеллы XIX века."},
    {"code": "LIT-203", "author": "H.G.Wells",
     "name": "Научная фантастика: стиль Г. Уэллса",
     "blurb": "Ранняя научная фантастика и социальная сатира."},
    {"code": "LIT-204", "author": "JackLondon",
     "name": "Приключенческая проза: стиль Д. Лондона",
     "blurb": "Человек против природы, суровый реализм Севера."},
    {"code": "LIT-205", "author": "MarkTwain",
     "name": "Сатирическая проза: стиль М. Твена",
     "blurb": "Разговорный American English и ирония."},
]
SUBJECTS_BY_CODE = {s["code"]: s for s in SUBJECTS}


def tier_of(pct: float) -> str:
    if pct <= BAND_RED_MAX:
        return "red"
    if pct >= BAND_GREEN_MIN:
        return "green"
    return "yellow"


def _ai_tier(ai_pct: float) -> str:
    """Для AI% семантика цвета обратная относительно 'соответствия стилю':
    низкий AI% - хорошо (зелёный/оригинальный текст), высокий - тревожно
    (красный/вероятно ИИ). Пороги здесь СВОИ (AI_BAND_*), отдельные от
    порогов совпадения стиля - у ложного обвинения в ИИ намного выше цена
    ошибки, чем у заниженного Shyndyq %, поэтому по умолчанию AI-порог
    красного заметно консервативнее (см. AI_BAND_RED_MAX/AI_BAND_GREEN_MIN
    выше в файле)."""
    if ai_pct >= AI_BAND_GREEN_MIN:
        return "red"
    if ai_pct <= AI_BAND_RED_MAX:
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
        if cur_words >= WINDOW_TARGET_WORDS:
            flush()
    flush()

    # Последнее окно может быть слишком маленьким - сливаем с предыдущим.
    if len(windows) >= 2 and windows[-1]["word_count"] < WINDOW_MIN_WORDS:
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
        meta_path = MODEL_DIR / "model_meta.json"
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


def analyze_text(raw_text: str, target_author: str) -> dict:
    if not raw_text or not raw_text.strip():
        raise ValueError("Текст пуст или не удалось извлечь текст из файла.")

    author_classes = list(LABEL_ENCODER.classes_)
    if target_author not in author_classes:
        raise ValueError(f"Неизвестный автор: {target_author}")
    target_idx = author_classes.index(target_author)

    # ---- 1) Официальный документ-уровневый вердикт (как в CLI predict) ----
    doc_result = predict_author(raw_text, model_dir=MODEL_DIR)
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
    proba = PIPELINE.predict_proba(window_texts)  # (n_windows, n_authors)
    for i, w in enumerate(windows):
        w["style_score"] = float(proba[i][target_idx])

    # --- независимый расчёт №1b: детектор "вне распределения" (novelty) ---
    # Отвечает не "какой автор ближе", а "похож ли текст вообще на корпус,
    # на котором обучалась стилевая модель" - см. train_model.py. Если
    # OOD_DETECTOR отсутствует (старая модель без ood_novelty.joblib),
    # window_novelty будет None и весь дальнейший код должен это учитывать.
    window_novelty = None
    if OOD_DETECTOR is not None:
        window_novelty = train_model.novelty_pct_scores(window_texts, PIPELINE, OOD_DETECTOR)
        for i, w in enumerate(windows):
            w["novelty_pct"] = round(float(window_novelty[i]), 1)

    # --- независимый расчёт №2: обученный классификатор "человек/ИИ",
    #     с откатом на эвристику для слишком коротких фрагментов или если
    #     модель не обучена (см. train_ai_detector.py) ---
    ai_sources_used = set()
    for w in windows:
        res = ai_detector.score_fragment(w["text"], PROJECT_DIR)
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
    style_reliable = (not NOVELTY_GATING_ENABLED) or novelty_pct is None \
        or novelty_pct >= NOVELTY_UNRELIABLE_MAX

    return {
        "target_author": target_author,
        "target_author_display": AUTHOR_DISPLAY_NAMES.get(target_author, target_author),
        "style_overall_pct": style_overall_pct,
        "style_tier": tier_of(style_overall_pct),
        "style_predicted_author": doc_result["predicted_author"],
        "style_predicted_author_display": AUTHOR_DISPLAY_NAMES.get(
            doc_result["predicted_author"], doc_result["predicted_author"]),
        "style_ranking": [
            {"author": AUTHOR_DISPLAY_NAMES.get(r["author"], r["author"]),
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


# =============================================================================
# Хранилище сданных работ - постоянное (SQLite, см. storage.py), переживает
# перезапуск процесса. Путь к файлу базы настраивается через переменную
# окружения SHYNDYQ_DB_PATH (удобно для тестов - см. tests/conftest.py) или
# по умолчанию лежит рядом с app.py.
# =============================================================================

DB_PATH = Path(os.environ.get("SHYNDYQ_DB_PATH", str(PROJECT_DIR / "shyndyq.db")))
SUBMISSIONS = storage.SubmissionStore(DB_PATH)


def make_submission(title: str, subject_code: str, raw_text: str,
                     sub_date: str) -> dict:
    subject = SUBJECTS_BY_CODE[subject_code]
    target_author = subject["author"]
    result = analyze_text(raw_text, target_author)
    style_v, style_stamp, style_stamp_cls = style_verdict_text(
        result["style_overall_pct"], result["style_tier"],
        result["target_author_display"],
        result["style_predicted_author_display"],
        result["style_predicted_author"] == result["target_author"],
        ai_tier=result["ai_tier"],
        novelty_pct=result["novelty_pct"],
        style_reliable=result["style_reliable"])
    ai_v, ai_stamp, ai_stamp_cls = ai_verdict_text(
        result["ai_overall_pct"], result["ai_tier"])

    sub_id = SUBMISSIONS.next_id()
    sub = {
        "id": sub_id,
        "title": title or f"Работа №{sub_id}",
        "subject_code": subject_code,
        "subject_name": subject["name"],
        "date": sub_date,
        **result,
        "style_verdict": style_v,
        "style_stamp": style_stamp,
        "style_stamp_class": style_stamp_cls,
        "ai_verdict": ai_v,
        "ai_stamp": ai_stamp,
        "ai_stamp_class": ai_stamp_cls,
    }
    SUBMISSIONS[sub_id] = sub
    return sub


def _truncate_at_paragraph(text: str, target_words: int) -> str:
    """Обрезает текст до ~target_words слов, останавливаясь на границе
    абзаца (пустая строка), а не разрывая его - в отличие от split()+join()
    по словам, это сохраняет структуру абзацев, важную для стилометрических
    признаков (длина абзаца, доля диалога и т.д.)."""
    blocks = re.split(r"(\n\s*\n)", text)
    out, words_so_far = [], 0
    for part in blocks:
        out.append(part)
        words_so_far += len(part.split())
        if words_so_far >= target_words:
            break
    return "".join(out).strip()


def seed_demo_submissions() -> None:
    """Заполняет платформу несколькими реально проанализированными работами,
    чтобы разбор можно было показать сразу, без загрузки файлов."""
    data_dir = Path(__file__).parent / "data"

    doyle_path = data_dir / "ArthurConanDoyle" / "Book_1.txt"
    if doyle_path.exists():
        text = doyle_path.read_text(encoding="utf-8", errors="replace")
        excerpt = _truncate_at_paragraph(text, 1500)
        make_submission("Возвращение на Бейкер-стрит (отрывок).docx",
                         "LIT-201", excerpt, "12 марта")

    # ВАЖНО: раньше этот файл лежал в Tairlan/Book_1.txt (в корне репозитория,
    # а не в data/) - путь ниже никогда не совпадал ни разу с самого первого
    # коммита проекта, поэтому третья демо-работа никогда фактически не
    # создавалась. Перенесено в data/_demo_samples/ с понятным именем при
    # чистке архитектуры - содержимое specifically подобрано так, чтобы
    # демонстрировать сценарий "стиль не похож на автора, но это не ИИ"
    # (см. mismatch-баннер в Shyndyq.jsx/report.html).
    poe_path = data_dir / "_demo_samples" / "gothic_essay_style_mismatch_sample.txt"
    if poe_path.exists():
        text = poe_path.read_text(encoding="utf-8", errors="replace")
        make_submission("Эссе о родоначальнике готической литературы.pdf",
                         "LIT-202", text, "15 марта")

    if doyle_path.exists():
        text = doyle_path.read_text(encoding="utf-8", errors="replace")
        excerpt = _truncate_at_paragraph(text, 900)
        ai_insert = (
            "In today's world, it is important to note that this comprehensive "
            "analysis will delve into the intricate tapestry of the detective's "
            "reasoning. Moreover, the narrative seeks to showcase a multifaceted "
            "approach to justice. Furthermore, one must navigate the complex "
            "landscape of Victorian society. Additionally, this endeavor will "
            "foster a deeper understanding of the case. In conclusion, the story "
            "underscores a robust and seamless exploration of these themes, "
            "highlighting the pivotal role that observation plays in the pursuit "
            "of truth and the unwavering commitment to justice that defines the "
            "detective's methodology throughout this comprehensive narrative."
        )
        combined = excerpt + "\n\n" + ai_insert
        make_submission("Продолжение рассказа о Шерлоке Холмсе.docx",
                         "LIT-201", combined, "18 марта")


# =============================================================================
# Фоновые задания анализа (имитация длительной обработки: студенту показывают
# анимацию стадий, пока в отдельном потоке реально считается результат).
# =============================================================================

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _run_job(job_id: str, title: str, subject_code: str, raw_text: str) -> None:
    try:
        sub = make_submission(title, subject_code, raw_text,
                               date.today().strftime("%d.%m.%Y"))
        with JOBS_LOCK:
            JOBS[job_id] = {"status": "done", "sub_id": sub["id"]}
    except ValueError as e:
        with JOBS_LOCK:
            JOBS[job_id] = {"status": "error", "error": str(e)}
    except Exception as e:  # noqa: BLE001
        with JOBS_LOCK:
            JOBS[job_id] = {"status": "error", "error": f"Ошибка анализа: {e}"}


# =============================================================================
# Flask routes
# =============================================================================

@app.route("/")
def root():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    subs = sorted(SUBMISSIONS.values(), key=lambda s: s["id"], reverse=True)
    return render_template("dashboard.html", submissions=subs, subjects=SUBJECTS)


@app.route("/submit-work")
def submit_page():
    preselect = request.args.get("subject", SUBJECTS[0]["code"])
    return render_template("submit.html", subjects=SUBJECTS, preselect=preselect)


@app.route("/submit", methods=["POST"])
def submit():
    subject_code = request.form.get("subject", "")
    title = request.form.get("title", "").strip()
    text = request.form.get("text", "")

    if subject_code not in SUBJECTS_BY_CODE:
        return jsonify({"error": "Выберите предмет."}), 400

    upload = request.files.get("file")
    if upload and upload.filename:
        raw_bytes = upload.read()
        try:
            text = doc_extract.extract_text(upload.filename, raw_bytes)
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": f"Не удалось прочитать файл: {e}"}), 400
        if not title:
            title = upload.filename

    if not text or not text.strip():
        return jsonify({"error": "Не удалось получить текст работы — загрузите файл или вставьте текст."}), 400

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "processing"}
    thread = threading.Thread(target=_run_job, args=(job_id, title, subject_code, text), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/job-status/<job_id>")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"status": "error", "error": "Задание не найдено."}), 404
    payload = {"status": job["status"]}
    if job["status"] == "done":
        payload["redirect"] = url_for("dashboard") + f"#work-{job['sub_id']}"
    elif job["status"] == "error":
        payload["error"] = job["error"]
    return jsonify(payload)


@app.route("/report/<int:sub_id>")
def report(sub_id: int):
    sub = SUBMISSIONS.get(sub_id)
    if sub is None:
        abort(404)
    mode = request.args.get("mode", "style")
    if mode not in ("style", "ai"):
        mode = "style"
    return render_template("report.html", sub=sub, initial_mode=mode)


@app.route("/report/<int:sub_id>/simple")
def report_simple(sub_id: int):
    sub = SUBMISSIONS.get(sub_id)
    if sub is None:
        abort(404)
    return render_template("report_simple.html", sub=sub)


def main():
    global PIPELINE, LABEL_ENCODER, OOD_DETECTOR, MODEL_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=Path("./model"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    MODEL_DIR = args.model_dir
    print(f"Пороги стиля: red_max={BAND_RED_MAX} / green_min={BAND_GREEN_MIN}")
    print(f"Пороги AI Detection: red_max={AI_BAND_RED_MAX} / green_min={AI_BAND_GREEN_MIN}")
    print(f"Загрузка модели из {MODEL_DIR} ...")
    PIPELINE, LABEL_ENCODER = load_model(MODEL_DIR)
    OOD_DETECTOR = train_model.load_ood_detector(MODEL_DIR)
    if OOD_DETECTOR is None:
        print("ВНИМАНИЕ: ood_novelty.joblib не найден - детектор новизны "
              "отключён, style-вердикт шлюзуется только по AI%. "
              "Переобучите модель (train_model.py train), чтобы получить "
              "также защиту от статистически аномального (не только "
              "ИИ-специфичного) текста.")
    print(f"Готово. Авторы: {list(LABEL_ENCODER.classes_)}")
    # Сеем демо-данные, только если хранилище пустое (первый запуск) - теперь,
    # когда SUBMISSIONS - постоянное SQLite-хранилище (см. storage.py), а не
    # словарь в памяти, повторный вызов при КАЖДОМ перезапуске плодил бы
    # дубликаты демо-работ поверх уже сохранённых настоящих сдач.
    if len(SUBMISSIONS) == 0:
        print("Хранилище пустое - готовлю демонстрационные работы...")
        seed_demo_submissions()
        print(f"Демо-работ загружено: {len(SUBMISSIONS)}")
    else:
        print(f"Хранилище уже содержит {len(SUBMISSIONS)} сдач(и) - демо-данные не досеваются.")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
