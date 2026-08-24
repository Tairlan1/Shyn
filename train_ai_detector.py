#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ai_detector.py
=====================================================================
Обучает классификатор "написано человеком / похоже на текст ИИ" на
контрастном датасете:

  - класс "AI": data_ai_detector/ai_written.txt - ~70 отрывков,
    написанных напрямую языковой моделью (Claude) по большому числу
    разных тем и регистров (наука, здоровье, бизнес, личные эссе,
    путешествия, экология, образование, психология, финансы и т.д.).
    Это единственный практически доступный источник ГАРАНТИРОВАННО
    ИИ-сгенерированного текста без выхода в интернет за чужими
    моделями или датасетами.
  - класс "человек": сбалансированная выборка из нескольких источников
    подлинно человеческого текста - корпус пяти авторов проекта
    (data_processed/dataset.jsonl), а также nltk brown/reuters/gutenberg
    (пресса, эссе, художественная проза, философия, поэзия и т.д.), для
    жанрового разнообразия за пределами прозы XIX века.

Признаки: TF-IDF по символьным и словесным n-граммам + поверхностные
стилометрические признаки (ai_features.SurfaceFeaturizer). Модель -
логистическая регрессия с L2-регуляризацией (архитектура сознательно
проще, чем у author_style_pipeline: датасет на порядок меньше, поэтому
переусложнение модели дало бы переобучение, а не точность).

ЧЕСТНАЯ ОЦЕНКА ОГРАНИЧЕНИЙ (обязательно прочитай перед тем как
опираться на эту модель в проде):
  - Класс "AI" целиком написан ОДНОЙ моделью (Claude) в её естественном
    стиле письма по запросу. Реальные тексты, сгенерированные другими
    моделями (ChatGPT, Gemini, локальные LLM) или через промпты,
    имитирующие человеческий стиль, могут иметь заметно другие
    поверхностные признаки - обобщение на них не гарантировано.
  - Обучающая выборка небольшая (порядка 150 примеров) по меркам
    машинного обучения. Кросс-валидация даёт честную оценку точности
    НА ЭТОМ распределении данных, но не является доказательством
    точности на реальных студенческих работах с примесью ИИ-правки.
  - Это не замена профессиональным коммерческим AI-детекторам, а
    прозрачная, полностью открытая и объяснимая альтернатива с чётко
    описанными границами применимости.

Запуск:
    python train_ai_detector.py

Результат:
    model_ai_detector/ai_detector_pipeline.joblib
    model_ai_detector/ai_detector_meta.json
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from ai_features import SurfaceFeaturizer

PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "model_ai_detector"
TARGET_WORDS = 190
MIN_WORDS = 100
RANDOM_SEED = 42


# =============================================================================
# Загрузка класса "AI"
# =============================================================================

def load_ai_passages() -> list[str]:
    path = PROJECT_DIR / "data_ai_detector" / "ai_written.txt"
    raw = path.read_text(encoding="utf-8")
    passages = [p.strip() for p in raw.split("===")]
    return [p for p in passages if len(p.split()) >= 40]


# =============================================================================
# Загрузка класса "человек" - несколько источников для жанрового разнообразия
# =============================================================================

def _chunk_sentences(sentences: list[str], target_words: int) -> list[str]:
    """Группирует список предложений в текстовые блоки ~target_words слов."""
    chunks, cur, cur_words = [], [], 0
    for s in sentences:
        cur.append(s)
        cur_words += len(s.split())
        if cur_words >= target_words:
            chunks.append(" ".join(cur))
            cur, cur_words = [], 0
    if cur and cur_words >= MIN_WORDS:
        chunks.append(" ".join(cur))
    return chunks


def load_human_from_nltk(corpus_name: str, max_chunks: int) -> list[str]:
    import nltk
    try:
        nltk.data.find(f"corpora/{corpus_name}")
    except LookupError:
        nltk.download(corpus_name, quiet=True)

    from nltk.corpus import brown, gutenberg, reuters
    corpus = {"brown": brown, "gutenberg": gutenberg, "reuters": reuters}[corpus_name]

    sentences = []
    for sent_tokens in corpus.sents():
        sent = " ".join(sent_tokens)
        sent = re.sub(r"\s+([.,;:!?])", r"\1", sent)  # убрать пробел перед пунктуацией
        if len(sent.split()) >= 4:
            sentences.append(sent)

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(sentences)
    chunks = _chunk_sentences(sentences, TARGET_WORDS)
    rng.shuffle(chunks)
    return chunks[:max_chunks]


def load_human_passages(target_count: int) -> list[str]:
    import preprocess_corpus as prep

    per_source = max(6, target_count // 4)
    chunks: list[str] = []

    for corpus_name in ("brown", "reuters", "gutenberg"):
        chunks.extend(load_human_from_nltk(corpus_name, per_source))

    # проектный корпус (5 авторов) - нарезаем под-окна вручную здесь,
    # чтобы переиспользовать sentence-сплиттер проекта без циклического импорта
    dataset_path = PROJECT_DIR / "data_processed" / "dataset.jsonl"
    if dataset_path.exists():
        rows = []
        with dataset_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        rng = random.Random(RANDOM_SEED)
        rng.shuffle(rows)
        project_chunks = []
        for row in rows:
            sentences = prep.split_sentences(row["text"])
            project_chunks.extend(_chunk_sentences(sentences, TARGET_WORDS))
            if len(project_chunks) >= per_source * 3:
                break
        rng.shuffle(project_chunks)
        chunks.extend(project_chunks[:per_source])

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(chunks)
    return chunks[:target_count]


# =============================================================================
# Пайплайн признаков + модель
# =============================================================================

def _identity(x):
    return x


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), max_features=2500,
            sublinear_tf=True, min_df=2)),
        ("word_tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), max_features=1200,
            sublinear_tf=True, min_df=2, stop_words="english")),
        ("surface", SurfaceFeaturizer()),
    ])
    clf = LogisticRegression(
        max_iter=3000, C=0.7, class_weight="balanced", random_state=RANDOM_SEED)
    return Pipeline([("features", features), ("clf", clf)])


def main():
    print("Загружаю класс AI...")
    ai_texts = load_ai_passages()
    print(f"  {len(ai_texts)} отрывков")

    print("Загружаю класс 'человек' (brown/reuters/gutenberg/project corpus)...")
    human_texts = load_human_passages(target_count=len(ai_texts) + 15)
    print(f"  {len(human_texts)} отрывков")

    X = ai_texts + human_texts
    y = [1] * len(ai_texts) + [0] * len(human_texts)  # 1 = AI, 0 = человек

    print(f"\nВсего примеров: {len(X)} (AI={sum(y)}, человек={len(y)-sum(y)})")

    pipeline = build_pipeline()

    print("\nКросс-валидация (5-fold, честная оценка на неувиденных примерах)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, method="predict")
    y_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]

    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    print(f"\nAccuracy (CV): {acc:.3f}")
    print(f"ROC-AUC (CV): {auc:.3f}")
    print(classification_report(y, y_pred, target_names=["человек", "AI"]))

    print("Дообучаю финальную модель на 100% данных...")
    pipeline.fit(X, y)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "ai_detector_pipeline.joblib")

    meta = {
        "n_ai_examples": len(ai_texts),
        "n_human_examples": len(human_texts),
        "human_sources": ["brown", "reuters", "gutenberg", "project_5author_corpus"],
        "ai_source": "hand-authored by Claude across ~70 topics/registers",
        "cv_accuracy": round(float(acc), 4),
        "cv_roc_auc": round(float(auc), 4),
        "target_chunk_words": TARGET_WORDS,
        "limitations": (
            "AI class is single-model-authored (Claude); small dataset "
            "(~" + str(len(X)) + " examples); cross-validation accuracy reflects "
            "this specific data distribution, not real-world student submissions. "
            "Not a substitute for professional commercial AI detectors."
        ),
    }
    (MODEL_DIR / "ai_detector_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nСохранено в {MODEL_DIR}/")


if __name__ == "__main__":
    main()
