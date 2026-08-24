#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_model.py
=====================================================================
Обучает и применяет модель атрибуции авторства на датасете, созданном
`preprocess_corpus.py` (dataset.jsonl: author, book, text, ...).

Подход основан на классической стилометрии, которая на корпусах такого
масштаба (несколько авторов, десятки-сотни тысяч слов на автора) стабильно
превосходит дообучение больших нейросетей "с нуля" и является эталоном в
задачах Authorship Attribution (в т.ч. в конкурсах PAN@CLEF):

  - символьные n-граммы (char n-grams, TF-IDF) - главный сигнал, ловит
    орфографические, пунктуационные и морфологические привычки автора
    независимо от темы текста;
  - частоты служебных/функциональных слов (Burrows' Delta-подобный сигнал) -
    авторский "синтаксический скелет";
  - явные стилометрические признаки: длина предложений и её вариативность,
    богатство словаря (TTR, hapax legomena), пунктуация, доля диалога,
    структура абзацев.

Финальная модель - калиброванный стэкинг (soft-voting/stacking) нескольких
линейных классификаторов поверх этих признаков, что даёт как точный класс,
так и откалиброванную вероятность принадлежности каждому автору.

Обучение (документ-уровневая кросс-валидация, чтобы избежать утечки
данных между чанками одной и той же книги):

    python train_model.py train \
        --data /data_processed/dataset.jsonl \
        --model-dir /model

Инференс на новом (неизвестном) тексте:

    python train_model.py predict \
        --model-dir /model \
        --input /path/to/unknown_text.txt

    python train_model.py predict \
        --model-dir /model \
        --text "some raw text ..."
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                      StratifiedGroupKFold)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
import joblib

# Переиспользуем разбиение на предложения/абзацы и чанкинг из скрипта
# предобработки, чтобы инференс на "сыром" неизвестном тексте проходил
# через ТОТ ЖЕ пайплайн очистки/нарезки, что и обучающие данные.
import preprocess_corpus as prep

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_FILE = "author_style_pipeline.joblib"
LABELS_FILE = "label_encoder.joblib"
META_FILE = "model_meta.json"


# =============================================================================
# РУЧНЫЕ СТИЛОМЕТРИЧЕСКИЕ ПРИЗНАКИ
# =============================================================================

# Компактный, но информативный список английских служебных слов
# (предлоги, союзы, местоимения, вспомогательные глаголы, детерминаторы).
# Именно частоты ЭТИХ слов - основа классической атрибуции авторства
# (Burrows' Delta, Mosteller-Wallace) поскольку их употребление почти не
# зависит от темы произведения и сильно зависит от индивидуального стиля.
FUNCTION_WORDS = """
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over
own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't
would wouldn't you you'd you'll you're you've your yours yourself
yourselves upon whilst thou thee thy ye
""".split()
FUNCTION_WORDS = sorted(set(FUNCTION_WORDS))

_PUNCT_CHARS = [",", ";", ":", "-", "--", "!", "?", "...", '"', "'", "(", ")"]


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def extract_stylometric_features(text: str) -> dict:
    """Считает набор явных стилометрических признаков для одного чанка."""
    sentences = prep.split_sentences(text)
    words = [w for w in text.split() if any(c.isalpha() for c in w)]
    words_lower = [w.strip(".,;:!?\"'()").lower() for w in words]
    words_lower = [w for w in words_lower if w]
    n_words = max(1, len(words))
    n_sentences = max(1, len(sentences))
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    n_paragraphs = max(1, len(paragraphs))

    sent_lengths = [len(s.split()) for s in sentences] or [n_words]
    word_lengths = [len(w) for w in words_lower] or [0]

    vocab = set(words_lower)
    freq = {}
    for w in words_lower:
        freq[w] = freq.get(w, 0) + 1
    hapax = sum(1 for c in freq.values() if c == 1)

    feats = {
        "avg_sentence_len": float(np.mean(sent_lengths)),
        "std_sentence_len": float(np.std(sent_lengths)),
        "avg_word_len": float(np.mean(word_lengths)),
        "std_word_len": float(np.std(word_lengths)),
        "ttr": _safe_div(len(vocab), n_words),                 # type-token ratio
        "hapax_ratio": _safe_div(hapax, n_words),               # богатство словаря
        "avg_paragraph_len_words": _safe_div(n_words, n_paragraphs),
        "sentences_per_paragraph": _safe_div(n_sentences, n_paragraphs),
        "dialogue_paragraph_ratio": _safe_div(
            sum(1 for p in paragraphs if prep.is_dialogue_paragraph(p)), n_paragraphs),
        "comma_per_100w": 100 * _safe_div(text.count(","), n_words),
        "semicolon_per_100w": 100 * _safe_div(text.count(";"), n_words),
        "colon_per_100w": 100 * _safe_div(text.count(":"), n_words),
        "exclaim_per_100w": 100 * _safe_div(text.count("!"), n_words),
        "question_per_100w": 100 * _safe_div(text.count("?"), n_words),
        "dash_per_100w": 100 * _safe_div(text.count("--"), n_words),
        "quote_per_100w": 100 * _safe_div(text.count('"'), n_words),
        "capitalized_word_ratio": _safe_div(
            sum(1 for w in words if w[:1].isupper()), n_words),
        "long_word_ratio": _safe_div(
            sum(1 for w in words_lower if len(w) >= 7), n_words),
        "short_word_ratio": _safe_div(
            sum(1 for w in words_lower if len(w) <= 3), n_words),
    }
    for fw in FUNCTION_WORDS:
        feats[f"fw_{fw}"] = 1000 * _safe_div(freq.get(fw, 0), n_words)
    return feats


class StylometricFeaturizer(BaseEstimator, TransformerMixin):
    """sklearn-совместимый трансформер: текст -> вектор явных
    стилометрических признаков (для использования в Pipeline/FeatureUnion)."""

    def fit(self, X, y=None):
        sample = extract_stylometric_features(X[0] if len(X) else "")
        self.feature_names_ = sorted(sample.keys())
        return self

    def transform(self, X):
        rows = []
        for text in X:
            feats = extract_stylometric_features(text)
            rows.append([feats.get(name, 0.0) for name in self.feature_names_])
        return np.asarray(rows, dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_)


class DenseTransformer(BaseEstimator, TransformerMixin):
    """Приводит разреженную матрицу к плотной (нужно StandardScaler-у)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if sparse.issparse(X) else X


# =============================================================================
# ЗАГРУЗКА ДАННЫХ
# =============================================================================

def load_dataset(path: Path) -> tuple[list[str], list[str], list[str]]:
    """Возвращает (тексты, метки_автора, id_книги_для_группировки)."""
    texts, authors, groups = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            texts.append(rec["text"])
            authors.append(rec["author"])
            groups.append(f"{rec['author']}::{rec['book']}")
    return texts, authors, groups


# =============================================================================
# ПОСТРОЕНИЕ МОДЕЛИ
# =============================================================================

def _safe_inner_cv(y, requested: int = 3) -> int:
    """Внутренние cv (StackingClassifier, CalibratedClassifierCV) не могут
    запросить больше фолдов, чем самое малочисленное количество примеров
    какого-либо класса в данных. На больших корпусах это всегда просто
    вернёт `requested`; страхует только очень маленькие/несбалансированные
    датасеты."""
    if y is None or len(y) == 0:
        return 2
    counts = np.bincount(y)
    counts = counts[counts > 0]
    min_count = int(counts.min()) if len(counts) else 2
    # StackingClassifier само ещё раз делит train на `requested` частей перед
    # тем как каждый базовый классификатор увидит свою долю - закладываем
    # на это запас, чтобы вложенный CalibratedClassifierCV не запросил
    # больше фолдов, чем реально доступно примеров редкого класса.
    effective_min = max(1, min_count // 2)
    return max(2, min(requested, effective_min))


def build_pipeline(n_authors: int, inner_cv: int = 3) -> Pipeline:
    """Собирает финальный pipeline: FeatureUnion(char-ngram TF-IDF,
    word-ngram TF-IDF, стилометрия) -> стэкинг классификаторов."""

    char_tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5),
        min_df=2, max_df=0.95, sublinear_tf=True,
        max_features=60_000,
    )
    word_tfidf = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        min_df=2, max_df=0.9, sublinear_tf=True,
        max_features=30_000, lowercase=True,
    )
    stylo = Pipeline([
        ("features", StylometricFeaturizer()),
        ("scale", StandardScaler()),
    ])

    features = FeatureUnion([
        ("char_tfidf", char_tfidf),
        ("word_tfidf", word_tfidf),
        ("stylometric", stylo),
    ])

    # Базовые классификаторы, каждый из которых - известный сильный вариант
    # для TF-IDF-признаков в задачах атрибуции авторства.
    base_estimators = [
        ("linsvc", CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced", dual="auto"),
            method="sigmoid", cv=inner_cv)),
        ("logreg", LogisticRegression(
            C=2.0, max_iter=3000, class_weight="balanced")),
        ("sgd", CalibratedClassifierCV(
            SGDClassifier(loss="modified_huber", alpha=1e-5,
                           class_weight="balanced", max_iter=2000),
            method="sigmoid", cv=inner_cv)),
    ]
    final_estimator = LogisticRegression(max_iter=3000)

    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
        cv=inner_cv,
    )

    pipeline = Pipeline([
        ("features", features),
        ("clf", stacking),
    ])
    return pipeline


# =============================================================================
# ОБУЧЕНИЕ + ОЦЕНКА КАЧЕСТВА (группировка по книге против утечки данных)
# =============================================================================

def train(data_path: Path, model_dir: Path, cv_folds: int = 5,
          test_size: float = 0.15, random_state: int = 42) -> None:
    print(f"Загрузка датасета: {data_path}")
    texts, authors, groups = load_dataset(data_path)
    n = len(texts)
    if n == 0:
        print("Датасет пуст. Сначала запустите preprocess_corpus.py", file=sys.stderr)
        sys.exit(1)

    le = LabelEncoder()
    y = le.fit_transform(authors)
    n_authors = len(le.classes_)
    print(f"Загружено {n} чанков, {n_authors} авторов: {list(le.classes_)}")

    n_groups = len(set(groups))
    if n_groups < 2:
        print("Внимание: в датасете только одна книга/группа - "
              "документ-уровневая кросс-валидация невозможна.", file=sys.stderr)

    # ---- Финальный отложенный тест (группировка по книге, чтобы чанки
    # одной и той же книги не оказались одновременно в train и test;
    # стратификация по автору, чтобы даже авторы с малым числом книг
    # гарантированно попали и в train, и в test) ----
    n_splits_holdout = max(2, round(1 / test_size))
    n_splits_holdout = min(n_splits_holdout, n_groups) if n_groups >= 2 else 2
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits_holdout, shuffle=True,
                                     random_state=random_state)
        train_idx, test_idx = next(sgkf.split(texts, y, groups=groups))
    except ValueError:
        # На совсем маленьких/несбалансированных данных стратификация может
        # быть невозможна - откатываемся на обычный группированный сплит.
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                      random_state=random_state)
        train_idx, test_idx = next(splitter.split(texts, y, groups=groups))
    X_train = [texts[i] for i in train_idx]
    y_train = y[train_idx]
    g_train = [groups[i] for i in train_idx]
    X_test = [texts[i] for i in test_idx]
    y_test = y[test_idx]

    print(f"Train: {len(X_train)} чанков | Test (отложенный, другие книги): {len(X_test)} чанков")

    # ---- Кросс-валидация на train (тоже группированная по книге) ----
    n_splits = min(cv_folds, len(set(g_train)))
    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        cv_scores = []
        for fold, (tr_i, va_i) in enumerate(gkf.split(X_train, y_train, groups=g_train)):
            # безопасный запас: stacking сам ещё раз разбивает train на inner_cv
            # частей, поэтому берём дополнительный запас прочности на маленьких данных
            fold_cv = max(2, min(_safe_inner_cv(y_train[tr_i]), 3))
            try:
                pipe = build_pipeline(n_authors, inner_cv=fold_cv)
                pipe.fit([X_train[i] for i in tr_i], y_train[tr_i])
                preds = pipe.predict([X_train[i] for i in va_i])
            except ValueError as exc:
                print(f"  CV фолд {fold + 1}/{n_splits}: пропущен (недостаточно "
                      f"примеров класса в этой группировке) - {exc}")
                continue
            acc = accuracy_score(y_train[va_i], preds)
            f1 = f1_score(y_train[va_i], preds, average="macro")
            cv_scores.append((acc, f1))
            print(f"  CV фолд {fold + 1}/{n_splits}: accuracy={acc:.4f}  macro-F1={f1:.4f}")
        if cv_scores:
            accs, f1s = zip(*cv_scores)
            print(f"Кросс-валидация (группировка по книге): "
                  f"accuracy={np.mean(accs):.4f}±{np.std(accs):.4f}  "
                  f"macro-F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")
        else:
            print("Все фолды кросс-валидации пропущены (слишком мало данных на "
                  "класс в разбиении) - переходим к финальному обучению.")
    else:
        print("Недостаточно книг для группированной кросс-валидации - "
              "пропускаем этот шаг и сразу обучаем финальную модель.")

    # ---- Финальное обучение на всём train, оценка на отложенном test ----
    print("Обучение финальной модели на всех train-данных...")
    try:
        final_pipeline = build_pipeline(n_authors, inner_cv=_safe_inner_cv(y_train))
        final_pipeline.fit(X_train, y_train)
    except ValueError:
        print("  Недостаточно данных для калиброванного стэкинга - "
              "понижаем внутреннюю кросс-валидацию до минимума (cv=2).")
        final_pipeline = build_pipeline(n_authors, inner_cv=2)
        final_pipeline.fit(X_train, y_train)

    if len(X_test):
        preds = final_pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro", labels=np.arange(n_authors),
                       zero_division=0)
        print(f"\nОтложенный тест (книги, не виденные при обучении): "
              f"accuracy={acc:.4f}  macro-F1={f1:.4f}\n")
        all_labels = np.arange(n_authors)
        print(classification_report(y_test, preds, labels=all_labels,
                                     target_names=le.classes_, digits=4,
                                     zero_division=0))
        cm = confusion_matrix(y_test, preds, labels=all_labels)
        print("Confusion matrix (строки=истина, столбцы=предсказание):")
        print("            " + "  ".join(f"{c[:10]:>10}" for c in le.classes_))
        for cls_name, row in zip(le.classes_, cm):
            print(f"{cls_name[:10]:>10}  " + "  ".join(f"{v:>10d}" for v in row))
        missing = [le.classes_[i] for i in all_labels if i not in set(y_test)]
        if missing:
            print(f"\nВнимание: в отложенном тесте не оказалось ни одной книги "
                  f"авторов {missing} (маленькая группа при группированном "
                  f"разбиении по книгам) - для них метрики выше посчитаны как 0 "
                  f"по поддержке (support=0), это не ошибка модели, а особенность "
                  f"конкретного разбиения. Кросс-валидация выше эту группу всё "
                  f"равно покрывает.")

    # ---- Дообучение на 100% данных для продакшн-модели ----
    print("\nДообучение финальной модели на 100% данных (для использования в проде)...")
    try:
        production_pipeline = build_pipeline(n_authors, inner_cv=_safe_inner_cv(y))
        production_pipeline.fit(texts, y)
    except ValueError:
        print("  Недостаточно данных для калиброванного стэкинга - "
              "понижаем внутреннюю кросс-валидацию до минимума (cv=2).")
        production_pipeline = build_pipeline(n_authors, inner_cv=2)
        production_pipeline.fit(texts, y)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(production_pipeline, model_dir / MODEL_FILE)
    joblib.dump(le, model_dir / LABELS_FILE)
    meta = {
        "authors": list(le.classes_),
        "n_chunks_trained_on": n,
        "held_out_accuracy": float(acc) if len(X_test) else None,
        "held_out_macro_f1": float(f1) if len(X_test) else None,
        "target_words_per_chunk": "1450-1550 (см. preprocess_corpus.py)",
    }
    (model_dir / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"\nМодель сохранена в: {model_dir / MODEL_FILE}")
    print(f"Метаданные: {model_dir / META_FILE}")


# =============================================================================
# ИНФЕРЕНС НА НОВОМ ТЕКСТЕ
# =============================================================================

def load_model(model_dir: Path):
    pipeline = joblib.load(model_dir / MODEL_FILE)
    le = joblib.load(model_dir / LABELS_FILE)
    return pipeline, le


def predict_author(text: str, model_dir: Path,
                    chunk_cfg: prep.ChunkingConfig | None = None) -> dict:
    """Определяет автора произвольного (неизвестного) текста.

    Текст прогоняется через ТОТ ЖЕ пайплайн очистки и нарезки на окна
    ~1500 слов, что и обучающие данные, затем модель предсказывает
    вероятности по каждому окну, и они агрегируются (усреднение
    логарифмов вероятностей = геометрическое среднее) в единый вердикт
    по всему тексту - это надёжнее, чем решение по одному окну.
    """
    pipeline, le = load_model(model_dir)
    cfg = chunk_cfg or prep.ChunkingConfig()

    cleaned = prep.clean_raw_text(text)
    paragraphs = prep.build_paragraph_index(cleaned)
    if not paragraphs:
        raise ValueError("После очистки текст оказался пустым - нечего анализировать.")

    chunks = prep.chunk_paragraphs(paragraphs, cfg)
    chunk_texts = [c["text"] for c in chunks]

    proba = pipeline.predict_proba(chunk_texts)  # (n_chunks, n_authors)
    log_proba = np.log(np.clip(proba, 1e-12, 1.0))
    mean_log_proba = log_proba.mean(axis=0)
    agg_proba = np.exp(mean_log_proba)
    agg_proba = agg_proba / agg_proba.sum()

    order = np.argsort(-agg_proba)
    ranked = [
        {"author": le.classes_[i], "probability": round(float(agg_proba[i]), 4)}
        for i in order
    ]

    per_chunk = []
    for i, ct in enumerate(chunk_texts):
        p = proba[i]
        top = int(np.argmax(p))
        per_chunk.append({
            "chunk_id": i,
            "word_count": chunks[i]["word_count"],
            "predicted_author": le.classes_[top],
            "confidence": round(float(p[top]), 4),
        })

    return {
        "predicted_author": ranked[0]["author"],
        "confidence": ranked[0]["probability"],
        "ranking": ranked,
        "n_chunks_analyzed": len(chunk_texts),
        "per_chunk_predictions": per_chunk,
    }


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Обучить модель на dataset.jsonl")
    p_train.add_argument("--data", type=Path, required=True,
                          help="Путь к dataset.jsonl (результат preprocess_corpus.py)")
    p_train.add_argument("--model-dir", type=Path, default=Path("./model"))
    p_train.add_argument("--cv-folds", type=int, default=5)
    p_train.add_argument("--test-size", type=float, default=0.15)

    p_pred = sub.add_parser("predict", help="Определить автора нового текста")
    p_pred.add_argument("--model-dir", type=Path, default=Path("./model"))
    group = p_pred.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Путь к .txt файлу неизвестного текста")
    group.add_argument("--text", type=str, help="Текст напрямую строкой")
    p_pred.add_argument("--json", action="store_true",
                         help="Вывести полный результат в формате JSON")

    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.command == "train":
        train(args.data, args.model_dir, cv_folds=args.cv_folds,
              test_size=args.test_size)

    elif args.command == "predict":
        raw_text = args.input.read_text(encoding="utf-8", errors="replace") \
            if args.input else args.text
        result = predict_author(raw_text, args.model_dir)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"\nПредсказанный автор: {result['predicted_author']} "
                  f"(уверенность {result['confidence']*100:.1f}%)")
            print(f"Проанализировано окон: {result['n_chunks_analyzed']}\n")
            print("Полное распределение вероятностей по авторам:")
            for r in result["ranking"]:
                bar = "#" * int(r["probability"] * 40)
                print(f"  {r['author']:<20} {r['probability']*100:6.2f}%  {bar}")


if __name__ == "__main__":
    main()