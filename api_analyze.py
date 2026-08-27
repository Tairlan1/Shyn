#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
api_analyze.py
=====================================================================
Настоящий REST-эндпоинт для интеграции с UniPlatform (или любым другим
внешним фронтендом). В отличие от app.py (который рендерит HTML-шаблоны),
здесь — только JSON, и анализ идёт по-абзацно, чтобы можно было
подсветить в интерфейсе, какие именно куски текста совпадают со стилем
студента/автора, а какие похожи на ИИ-генерацию (в том числе внутри
ОДНОГО файла, где текст намеренно смешан).

Ничего не предвычислено и не захардкожено: каждый вызов /api/v1/analyze
реально прогоняет присланный текст через
  - model/author_style_pipeline.joblib   (стиль)
  - model_ai_detector/ai_detector_pipeline.joblib  (ИИ-детектор,
    с откатом на ai_heuristics.py для слишком коротких абзацев)

Запуск:
    pip install flask-cors --break-system-packages   # если ещё не стоит
    python3 api_analyze.py
Слушает на 127.0.0.1:5001, CORS открыт для локальной разработки
(http://localhost:5173 — дефолтный порт Vite).
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

from flask import Flask, jsonify, request

import ai_detector
import ai_heuristics
import doc_extract
import preprocess_corpus as prep
import train_model as tm

PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "model"  # переопределяется флагом --model-dir при запуске

# author_style_pipeline.joblib был сохранён при запуске train_model.py
# напрямую (модуль __main__), поэтому кастомные трансформеры должны быть
# видны под именем __main__.* при анлоаде — иначе joblib.load падает.
sys.modules["__main__"].StylometricFeaturizer = tm.StylometricFeaturizer
sys.modules["__main__"].DenseTransformer = tm.DenseTransformer

app = Flask(__name__)

try:
    from flask_cors import CORS
    CORS(app)
except ImportError:
    @app.after_request
    def _add_cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return resp

MIN_WORDS_FOR_STYLE = 25  # значение по умолчанию для "старой" (только 1500 слов)
                           # модели - см. _load_style_model(), где оно
                           # автоматически понижается для multiscale-модели.

# Пороги СИНХРОНИЗИРОВАНЫ с app.py (BAND_RED_MAX/GREEN_MIN, AI_BAND_RED_MAX/
# GREEN_MIN) - раздельно для стиля и для ИИ, т.к. у ложного обвинения в ИИ
# цена ошибки намного выше, чем у заниженного style%. Меняя эти числа,
# обязательно меняйте и соответствующие в app.py/config - иначе Flask-версия
# (report.html) и это демо (UniPlatform) снова разойдутся в вердиктах.
STYLE_RED_MAX = 0.33
STYLE_GREEN_MIN = 0.80
AI_RED_MAX = 0.25
AI_GREEN_MIN = 0.90

_PIPELINE = None
_LABEL_ENCODER = None


def _load_style_model():
    global _PIPELINE, _LABEL_ENCODER, MIN_WORDS_FOR_STYLE
    if _PIPELINE is None:
        _PIPELINE, _LABEL_ENCODER = tm.load_model(MODEL_DIR)

        # Порог "текст слишком короткий для стилометрии" был откалиброван
        # под старую модель (обучена только на ~1500-словных чанках - при
        # 25 словах она НИКОГДА не видела ничего похожего на обучении, но
        # 25 всё равно было условной цифрой "хоть что-то", не измеренной).
        # Для multiscale-модели (см. train_model.py build-multiscale и
        # compare_models.py) есть честные held-out метрики по масштабам в
        # model_meta.json - "phrase" (10-25 слов) даёт ~60%+ accuracy на
        # НЕВИДЕННЫХ данных (при случайности 20% на 5 авторов), поэтому для
        # такой модели порог можно и нужно понизить, иначе преимущество
        # переобучения не используется на практике.
        meta_path = MODEL_DIR / "model_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("held_out_metrics_by_scale"):
                    MIN_WORDS_FOR_STYLE = 10
                    print(f"Обнаружена multiscale-модель ({MODEL_DIR}) - "
                          f"MIN_WORDS_FOR_STYLE понижен до {MIN_WORDS_FOR_STYLE} "
                          f"(см. held_out_metrics_by_scale в {meta_path.name}).")
            except (json.JSONDecodeError, OSError):
                pass
    return _PIPELINE, _LABEL_ENCODER


def _split_paragraphs(text: str) -> list[str]:
    cleaned = prep.clean_raw_text(text)
    paragraphs = [p.strip() for p in cleaned.split("\n\n")]
    return [p for p in paragraphs if p]


def _style_tier(style_score: float) -> str:
    if style_score <= STYLE_RED_MAX:
        return "red"
    if style_score >= STYLE_GREEN_MIN:
        return "green"
    return "yellow"


def _ai_tier(ai_score: float) -> str:
    """Семантика цвета обратная относительно стиля: низкий AI-score - хорошо
    (зелёный), высокий - тревожно (красный). Пороги намеренно асимметричны
    (red начинается только с 0.90) - см. комментарий у AI_GREEN_MIN выше."""
    if ai_score >= AI_GREEN_MIN:
        return "red"
    if ai_score <= AI_RED_MAX:
        return "green"
    return "yellow"


def _analyze_paragraph(paragraph: str, expected_author: str, pipeline, le) -> dict:
    words = [w for w in paragraph.split() if any(c.isalpha() for c in w)]
    word_count = len(words)

    entry = {
        "text": paragraph,
        "wordCount": word_count,
        "styleScore": None,
        "styleTier": None,  # 'red' | 'yellow' | 'green' | None (слишком короткий фрагмент)
        "topAuthor": None,
        "topAuthorProb": None,
        "aiScore": None,
        "aiTier": None,
        "aiSource": None,
    }

    # --- ИИ-детектор (работает от ~40 слов, иначе эвристика-резерв) ---
    ai_result = ai_detector.score_fragment(paragraph, PROJECT_DIR)
    if ai_result is not None:
        entry["aiScore"] = round(ai_result["ai_score"], 4)
        entry["aiSource"] = ai_result["source"]
    else:
        h = ai_heuristics.score_fragment(paragraph)
        entry["aiScore"] = round(h.get("ai_score", 0.0), 4)
        entry["aiSource"] = "heuristics_fallback"
    entry["aiTier"] = _ai_tier(entry["aiScore"])

    # --- Стилометрия (нужна более длинная выборка, иначе шумно) ---
    if word_count >= MIN_WORDS_FOR_STYLE:
        proba = pipeline.predict_proba([paragraph])[0]
        top_idx = int(proba.argmax())
        entry["topAuthor"] = le.classes_[top_idx]
        entry["topAuthorProb"] = round(float(proba[top_idx]), 4)
        if expected_author in le.classes_:
            exp_idx = list(le.classes_).index(expected_author)
            entry["styleScore"] = round(float(proba[exp_idx]), 4)
            entry["styleTier"] = _style_tier(entry["styleScore"])

    return entry


@app.route("/api/v1/health")
def health():
    return jsonify({"status": "ok", "model_dir": str(MODEL_DIR)})


@app.route("/api/v1/authors")
def authors():
    _, le = _load_style_model()
    return jsonify({"authors": list(le.classes_)})


@app.route("/api/v1/analyze", methods=["POST"])
def analyze():
    """
    Принимает multipart/form-data:
      - file: файл (.txt/.docx/.pdf) ИЛИ
      - text: сырой текст строкой
      - expected_author: ключ автора, с которым сверяем стиль (напр. "MarkTwain")
    Возвращает: агрегированный style%/ai% по документу + разбор по абзацам
    для подсветки в интерфейсе.
    """
    pipeline, le = _load_style_model()

    expected_author = request.form.get("expected_author", "")
    raw_text = request.form.get("text", "")

    if "file" in request.files and request.files["file"].filename:
        f = request.files["file"]
        raw_text = doc_extract.extract_text(f.filename, f.read())

    if not raw_text or not raw_text.strip():
        return jsonify({"error": "Пустой текст — нечего анализировать."}), 400

    paragraphs = _split_paragraphs(raw_text)
    if not paragraphs:
        return jsonify({"error": "После очистки текста не осталось содержимого."}), 400

    analyzed = [_analyze_paragraph(p, expected_author, pipeline, le) for p in paragraphs]

    scored = [a for a in analyzed if a["styleScore"] is not None]
    scored_ai = [a for a in analyzed if a["aiScore"] is not None]

    doc_style_score = round(sum(a["styleScore"] for a in scored) / len(scored), 4) if scored else None
    doc_ai_score = round(sum(a["aiScore"] for a in scored_ai) / len(scored_ai), 4) if scored_ai else 0.0

    doc_style_tier = _style_tier(doc_style_score) if doc_style_score is not None else None
    doc_ai_tier = _ai_tier(doc_ai_score)

    # ГЕЙТИНГ - та же логика, что в app.py/report.html: если ИИ-детектор
    # уверенно (красный уровень) считает текст сгенерированным, style% в
    # этом случае недостоверен вне зависимости от значения - фронтенд НЕ
    # должен показывать его как значимое совпадение (см. Shyndyq.jsx).
    style_reliable = doc_ai_tier != "red"

    flagged_paragraphs = sum(
        1 for a in analyzed
        if a["aiTier"] == "red" or (a["styleTier"] == "red")
    )

    return jsonify({
        "expectedAuthor": expected_author,
        "docStyleScore": doc_style_score,
        "docStyleTier": doc_style_tier,
        "docAiScore": doc_ai_score,
        "docAiTier": doc_ai_tier,
        "styleReliable": style_reliable,
        "totalParagraphs": len(analyzed),
        "flaggedParagraphs": flagged_paragraphs,
        "paragraphs": analyzed,
        "modelInfo": {
            "authors": list(le.classes_),
            "source": "author_style_pipeline.joblib + ai_detector_pipeline.joblib (реальный инференс, без предрасчёта)",
        },
    })


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR,
                     help="папка с author_style_pipeline.joblib "
                          "(например ./model_multiscale для новой модели)")
    ap.add_argument("--port", type=int, default=5001)
    args = ap.parse_args()

    MODEL_DIR = args.model_dir
    print(f"Загружаю модель из {MODEL_DIR} ...")
    _load_style_model()
    print(f"Модель загружена. Запускаю API на http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)
