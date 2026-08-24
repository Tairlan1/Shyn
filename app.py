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
import itertools
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
MODEL_DIR = Path("./model")
PROJECT_DIR = Path(__file__).parent

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

WINDOW_TARGET_WORDS = 180   # размер окна для подсветки фрагментов (компромисс
                             # между точностью модели, обученной на ~1500-словных
                             # окнах, и желаемой гранулярностью подсветки)
WINDOW_MIN_WORDS = 60

# Пороги цветовой классификации (в процентах) - единые для всей платформы:
# 0-33 красный / 33.01-79.99 жёлтый / 80-100 зелёный.
BAND_RED_MAX = 33.0
BAND_GREEN_MIN = 80.0


def tier_of(pct: float) -> str:
    if pct <= BAND_RED_MAX:
        return "red"
    if pct >= BAND_GREEN_MIN:
        return "green"
    return "yellow"


def _ai_tier(ai_pct: float) -> str:
    """Для AI% семантика цвета обратная относительно 'соответствия стилю':
    низкий AI% - хорошо (зелёный/оригинальный текст), высокий - тревожно
    (красный/вероятно ИИ). Используем те же границы 33/80 для единого
    визуального языка платформы."""
    if ai_pct >= BAND_GREEN_MIN:
        return "red"
    if ai_pct <= BAND_RED_MAX:
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
# Основной анализ (два НЕЗАВИСИМЫХ расчёта: стиль через ML-пайплайн,
# ИИ-индикатор через отдельную эвристику - друг на друга не влияют).
# =============================================================================

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
        "ai_overall_pct": round(ai_overall * 100, 1),
        "ai_tier": _ai_tier(round(ai_overall * 100, 1)),
        "ai_source_label": ai_source_label,
        "doc_html": doc_html,
        "n_windows": len(windows),
        "n_words": len(raw_text.split()),
    }


# =============================================================================
# Вердикты (детерминированные, на основе тех же порогов 33/80).
# =============================================================================

def style_verdict_text(style_pct: float, tier: str, target_display: str,
                        top_author_display: str, matches_top: bool) -> tuple[str, str, str]:
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
# Хранилище сданных работ (в памяти процесса - для демонстрации платформы;
# в реальном продукте это была бы таблица в БД учебной платформы).
# =============================================================================

SUBMISSIONS: dict[int, dict] = {}
_id_counter = itertools.count(1)


def make_submission(title: str, subject_code: str, raw_text: str,
                     sub_date: str) -> dict:
    subject = SUBJECTS_BY_CODE[subject_code]
    target_author = subject["author"]
    result = analyze_text(raw_text, target_author)
    style_v, style_stamp, style_stamp_cls = style_verdict_text(
        result["style_overall_pct"], result["style_tier"],
        result["target_author_display"],
        result["style_predicted_author_display"],
        result["style_predicted_author"] == result["target_author"])
    ai_v, ai_stamp, ai_stamp_cls = ai_verdict_text(
        result["ai_overall_pct"], result["ai_tier"])

    sub_id = next(_id_counter)
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

    poe_path = data_dir / "Tairlan" / "Book_1.txt"
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


def main():
    global PIPELINE, LABEL_ENCODER, MODEL_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=Path("./model"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    MODEL_DIR = args.model_dir
    print(f"Загрузка модели из {MODEL_DIR} ...")
    PIPELINE, LABEL_ENCODER = load_model(MODEL_DIR)
    print(f"Готово. Авторы: {list(LABEL_ENCODER.classes_)}")
    print("Готовлю демонстрационные работы...")
    seed_demo_submissions()
    print(f"Демо-работ загружено: {len(SUBMISSIONS)}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
