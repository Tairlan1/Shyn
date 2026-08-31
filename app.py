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
import os
import re
import threading
import uuid
from datetime import date
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for

import config
import doc_extract
import storage
import train_model
from analysis import (
    ai_verdict_text,
    analyze_text,
    style_verdict_text,
)
from config import PROJECT_DIR, SUBJECTS, SUBJECTS_BY_CODE
from train_model import load_model

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
    # создавалась. Перенесено в data/_demo_samples/ при чистке архитектуры.
    # Этот же файл используется автором репозитория для ручной проверки
    # AI-детектора на заведомо не участвовавшем в обучении тексте - поэтому
    # имя файла сохранено как есть ("Poel's original work.txt"), а не
    # переименовано.
    poe_path = data_dir / "_demo_samples" / "Poel's original work.txt"
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=Path("./model"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # ВАЖНО: пишем именно в config.* (а не в локальные/глобальные имена
    # этого модуля) - analysis.py читает состояние через `config.PIPELINE`
    # и т.д., и увидит эти значения только если они установлены НА САМОМ
    # ОБЪЕКТЕ МОДУЛЯ config, а не как одноимённые переменные здесь (см.
    # подробное объяснение в докстринге config.py).
    config.MODEL_DIR = args.model_dir
    print(f"Пороги стиля: red_max={config.BAND_RED_MAX} / green_min={config.BAND_GREEN_MIN}")
    print(f"Пороги AI Detection: red_max={config.AI_BAND_RED_MAX} / green_min={config.AI_BAND_GREEN_MIN}")
    print(f"Загрузка модели из {config.MODEL_DIR} ...")
    config.PIPELINE, config.LABEL_ENCODER = load_model(config.MODEL_DIR)
    config.OOD_DETECTOR = train_model.load_ood_detector(config.MODEL_DIR)
    if config.OOD_DETECTOR is None:
        print("ВНИМАНИЕ: ood_novelty.joblib не найден - детектор новизны "
              "отключён, style-вердикт шлюзуется только по AI%. "
              "Переобучите модель (train_model.py train), чтобы получить "
              "также защиту от статистически аномального (не только "
              "ИИ-специфичного) текста.")
    print(f"Готово. Авторы: {list(config.LABEL_ENCODER.classes_)}")
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
