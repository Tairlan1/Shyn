#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ai_detector.py
=====================================================================
Обучает классификатор "написано человеком / похоже на текст ИИ" на
контрастном датасете:

  - класс "AI": ПОДДЕРЖИВАЕТ НЕСКОЛЬКО ИСТОЧНИКОВ (см. load_ai_passages_by_source
    ниже) - data_ai_detector/sources/<имя_модели>/*.txt, каждый файл -
    отрывки, разделённые "===". Также поддерживается legacy-файл
    data_ai_detector/ai_written.txt (источник "claude_legacy") для
    обратной совместимости.
    НА МОМЕНТ НАПИСАНИЯ ЭТОГО КОДА в проекте есть только ОДИН источник
    (~70 отрывков от Claude) - структура готова принять данные от других
    моделей (GPT, Gemini, Llama и т.д.), когда они появятся, но их ещё
    физически нет. Это прямо влияет на то, чему можно доверять в отчёте
    ниже - см. "ЧЕСТНАЯ ОЦЕНКА ОГРАНИЧЕНИЙ".
  - класс "человек": сбалансированная выборка из нескольких источников
    подлинно человеческого текста - корпус пяти авторов проекта
    (data_processed/dataset.jsonl), а также nltk brown/reuters/gutenberg
    (пресса, эссе, художественная проза, философия, поэзия и т.д.), для
    жанрового разнообразия за пределами прозы XIX века.

Признаки: TF-IDF по символьным и словесным n-граммам + поверхностные
стилометрические признаки (ai_features.SurfaceFeaturizer, включая
структурные/ритмические сигналы - burstiness предложений и абзацев,
плотность markdown-разметки - специально выбранные как менее
модель-специфичные, чем лексикон конкретных клише). Модель - логисти-
ческая регрессия с L2-регуляризацией (архитектура сознательно проще,
чем у author_style_pipeline: датасет на порядок меньше, поэтому
переусложнение модели дало бы переобучение, а не точность).

ДВЕ РАЗНЫЕ ПРОВЕРКИ КАЧЕСТВА - НЕ ПУТАТЬ:
  1. Обычная 5-fold стратифицированная кросс-валидация (как раньше) -
     отвечает на вопрос "насколько хорошо модель разделяет ЭТИ конкретные
     примеры человека/ИИ". Высокая точность здесь ничего не говорит об
     обобщении на другие модели-генераторы, если весь класс "AI" из
     одного источника.
  2. Leave-one-source-out (LOSO) - если источников AI-текста 2 и больше,
     скрипт дополнительно обучает модель, ПОЛНОСТЬЮ исключив один
     источник (например, "gpt4"), и проверяет точность именно на нём -
     это честная имитация вопроса "как модель справится с генератором,
     которого не видела при обучении". Именно этот раздел отчёта -
     главный ориентир для доверия к обобщению на новые LLM, а не CV
     из пункта 1. При ОДНОМ источнике LOSO невозможен и явно помечается
     как "не измерено" - НЕ как "100% обобщение", а как отсутствие данных.

ЧЕСТНАЯ ОЦЕНКА ОГРАНИЧЕНИЙ (обязательно прочитай перед тем как
опираться на эту модель в проде):
  - Пока в data_ai_detector/sources/ лежит текст только ОДНОЙ модели -
    обобщение на другие LLM НЕ ИЗМЕРЕНО (не "хорошее" и не "плохое" -
    буквально неизвестно). Не выдавайте текущую cv_accuracy/cv_roc_auc
    за показатель качества на реальных студенческих работах с текстом от
    произвольной современной LLM.
  - Обучающая выборка небольшая (порядка 150 примеров) по меркам
    машинного обучения. Кросс-валидация даёт честную оценку точности
    НА ЭТОМ распределении данных, но не является доказательством
    точности на реальных студенческих работах с примесью ИИ-правки.
  - Это не замена профессиональным коммерческим AI-детекторам, а
    прозрачная, полностью открытая и объяснимая альтернатива с чётко
    описанными границами применимости.

КАК ДОБАВИТЬ НОВЫЙ ИСТОЧНИК (например, тексты от другой LLM):
    mkdir -p data_ai_detector/sources/gpt4
    # положите туда один или несколько .txt файлов; несколько отрывков
    # в одном файле разделяйте строкой "===" (как в ai_written.txt);
    # каждый отрывок >= 40 слов; старайтесь охватить разные темы,
    # регистры (формально/неформально), длину и, если возможно, разные
    # системные промпты/температуру, чтобы не переобучиться на одну
    # узкую "интонацию" конкретной модели.
    python train_ai_detector.py

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
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ai_features import SurfaceFeaturizer

PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "model_ai_detector"
SOURCES_DIR = PROJECT_DIR / "data_ai_detector" / "sources"
LEGACY_AI_FILE = PROJECT_DIR / "data_ai_detector" / "ai_written.txt"
TARGET_WORDS = 190
MIN_WORDS = 100
RANDOM_SEED = 42


# =============================================================================
# Загрузка класса "AI" - несколько источников (по одной папке на модель)
# =============================================================================

def _split_passages(raw: str) -> list[str]:
    passages = [p.strip() for p in raw.split("===")]
    return [p for p in passages if len(p.split()) >= 40]


def load_ai_passages_by_source() -> dict[str, list[str]]:
    """Возвращает {имя_источника: [отрывки]}. Источник - это подпапка в
    data_ai_detector/sources/<имя>/ (например "claude", "gpt4", "gemini",
    "llama"). Имя папки становится и меткой источника в метаданных, и
    именем "held-out" группы для leave-one-source-out проверки ниже.
    Legacy data_ai_detector/ai_written.txt (если существует) подключается
    как источник "claude_legacy" для обратной совместимости со старыми
    установками проекта."""
    sources: dict[str, list[str]] = {}

    if SOURCES_DIR.exists():
        for source_dir in sorted(p for p in SOURCES_DIR.iterdir() if p.is_dir()):
            passages = []
            for txt_file in sorted(source_dir.glob("*.txt")):
                passages.extend(_split_passages(txt_file.read_text(encoding="utf-8")))
            if passages:
                sources[source_dir.name] = passages

    if LEGACY_AI_FILE.exists():
        legacy_passages = _split_passages(LEGACY_AI_FILE.read_text(encoding="utf-8"))
        if legacy_passages:
            sources.setdefault("claude_legacy", []).extend(legacy_passages)

    return sources


def load_ai_passages() -> list[str]:
    """Плоский список всех AI-отрывков вне зависимости от источника -
    сохранён для обратной совместимости с кодом, которому источник не
    важен."""
    out = []
    for passages in load_ai_passages_by_source().values():
        out.extend(passages)
    return out


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

    from nltk.corpus import brown, gutenberg, reuters, webtext
    corpus = {"brown": brown, "gutenberg": gutenberg, "reuters": reuters,
              "webtext": webtext}[corpus_name]

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

    # ВАЖНО: brown/reuters/gutenberg - формальные регистры (журналистика
    # середины XX века, деловые новости, классика XIX века). Ни один из
    # них не похож на современный разговорный регистр, которым чаще всего
    # реально пишут студенты. webtext (отзывы, форумные посты, личные
    # объявления, обиходная речь) добавлен именно для этого - без него
    # эмпирически модель показывала завышенный P(AI) на обычном
    # современном неформальном тексте просто потому, что "человеческий"
    # класс никогда не видел ничего похожего на такой регистр.
    #
    # ПРОВЕРЕНО ЭМПИРИЧЕСКИ: при равных долях по 4 источникам webtext
    # оказывался слишком малой частью человеческого класса, чтобы
    # компенсировать перекос - признак contraction_per_100w (сокращения
    # вроде "didn't") в среднем оказывался ВЫШЕ у класса "AI" (1.12/100 слов),
    # чем у класса "человек" (0.28/100 слов), просто потому что три из
    # четырёх источников почти не используют сокращения в повествовании -
    # т.е. модель отчасти учится отличать "современный неформальный текст"
    # от "старая формальная проза", а не "ИИ" от "человек". Поэтому
    # webtext здесь взят с увеличенным весом (в 2 раза больше базовой
    # доли), чтобы контрастная выборка человека включала достаточно
    # современного разговорного регистра.
    per_source = max(6, target_count // 5)
    chunks: list[str] = []

    for corpus_name in ("brown", "reuters", "gutenberg"):
        chunks.extend(load_human_from_nltk(corpus_name, per_source))
    chunks.extend(load_human_from_nltk("webtext", per_source * 2))

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
        # ВАЖНО: SurfaceFeaturizer отдаёт признаки в СЫРЫХ единицах
        # разного масштаба (avg_sentence_len ~10-30, comma_per_100w ~0-20,
        # бинарные 0/1-сигналы), тогда как TF-IDF-векторы L2-нормализованы
        # (типичная компонента << 1). Без явного масштабирования линейная
        # модель непропорционально чувствительна именно к этому блоку -
        # эмпирически это приводило к завышенной и плохо откалиброванной
        # P(AI) даже на явно человеческом тексте. StandardScaler здесь
        # обязателен, а не косметика.
        ("surface", Pipeline([
            ("extract", SurfaceFeaturizer()),
            ("scale", StandardScaler()),
        ])),
    ])
    clf = LogisticRegression(
        max_iter=3000, C=0.3, class_weight="balanced", random_state=RANDOM_SEED)
    return Pipeline([("features", features), ("clf", clf)])


def main():
    print("Загружаю класс AI (по источникам)...")
    ai_by_source = load_ai_passages_by_source()
    if not ai_by_source:
        print(f"ОШИБКА: не найдено ни одного AI-примера ни в {SOURCES_DIR}, "
              f"ни в {LEGACY_AI_FILE}.")
        return
    ai_texts = [p for passages in ai_by_source.values() for p in passages]
    for name, passages in ai_by_source.items():
        print(f"  источник '{name}': {len(passages)} отрывков")
    print(f"  всего: {len(ai_texts)} отрывков из {len(ai_by_source)} источник(ов)")

    print("\nЗагружаю класс 'человек' (brown/reuters/gutenberg/project corpus)...")
    human_texts = load_human_passages(target_count=len(ai_texts) + 15)
    print(f"  {len(human_texts)} отрывков")

    X = ai_texts + human_texts
    y = [1] * len(ai_texts) + [0] * len(human_texts)  # 1 = AI, 0 = человек

    print(f"\nВсего примеров: {len(X)} (AI={sum(y)}, человек={len(y)-sum(y)})")

    pipeline = build_pipeline()

    print("\nКросс-валидация (5-fold, честная оценка на неувиденных примерах "
          "ИЗ ТЕХ ЖЕ источников AI-текста)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, method="predict")
    y_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]

    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    print(f"\nAccuracy (CV): {acc:.3f}")
    print(f"ROC-AUC (CV): {auc:.3f}")
    print(classification_report(y, y_pred, target_names=["человек", "AI"]))

    # ---- Leave-one-source-out (LOSO): честная проверка обобщения на
    # "невиденную" модель-генератор. Возможна только при 2+ источниках AI. ----
    loso_results = {}
    if len(ai_by_source) >= 2:
        print("\n" + "=" * 70)
        print("LEAVE-ONE-SOURCE-OUT: проверка обобщения на КАЖДЫЙ источник "
              "AI-текста, полностью исключённый из обучения")
        print("=" * 70)
        for held_out_name, held_out_passages in ai_by_source.items():
            train_ai = [p for name, passages in ai_by_source.items()
                        if name != held_out_name for p in passages]
            if not train_ai:
                continue
            X_loso = train_ai + human_texts
            y_loso = [1] * len(train_ai) + [0] * len(human_texts)
            loso_pipeline = build_pipeline()
            loso_pipeline.fit(X_loso, y_loso)

            X_held = held_out_passages
            y_held_pred = loso_pipeline.predict(X_held)
            held_acc = float(np.mean(y_held_pred == 1))  # доля верно найденных AI-примеров
            loso_results[held_out_name] = {
                "n_examples": len(X_held),
                "recall_on_unseen_source": round(held_acc, 4),
            }
            flag = "OK" if held_acc >= 0.7 else "СЛАБО"
            print(f"  источник '{held_out_name}' (n={len(X_held)}), НЕ участвовавший "
                  f"в обучении: обнаружено как AI {held_acc*100:.1f}% примеров  [{flag}]")
        print("Если recall для какого-то источника заметно ниже остальных - "
              "модель хуже обобщается именно на эту модель-генератор; "
              "стоит добавить больше примеров именно оттуда.")
    else:
        print("\nLEAVE-ONE-SOURCE-OUT: ПРОПУЩЕНО - в наличии только один "
              "источник AI-текста (" + next(iter(ai_by_source)) + "). "
              "Обобщение на другие модели-генераторы (GPT, Gemini, Llama и "
              "т.д.) НЕ ИЗМЕРЕНО - добавьте примеры от других моделей в "
              "data_ai_detector/sources/<имя_модели>/ и запустите заново, "
              "чтобы получить честную оценку (см. докстринг модуля).")

    print("\nДообучаю финальную модель на 100% данных...")
    pipeline.fit(X, y)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "ai_detector_pipeline.joblib")

    meta = {
        "n_ai_examples": len(ai_texts),
        "n_human_examples": len(human_texts),
        "ai_sources": {name: len(passages) for name, passages in ai_by_source.items()},
        "human_sources": ["brown", "reuters", "gutenberg", "webtext", "project_5author_corpus"],
        "cv_accuracy": round(float(acc), 4),
        "cv_roc_auc": round(float(auc), 4),
        "target_chunk_words": TARGET_WORDS,
        "leave_one_source_out": loso_results if loso_results else "not_measured_single_source",
        "limitations": (
            f"AI class currently spans {len(ai_by_source)} source(s): "
            f"{list(ai_by_source.keys())}. " +
            ("Generalization to other LLM families is NOT measured (single "
             "source) - do not treat cv_accuracy/cv_roc_auc as evidence of "
             "accuracy on arbitrary modern LLM output. "
             if len(ai_by_source) < 2 else
             "See leave_one_source_out for a per-source generalization "
             "estimate - still based on a small number of held-out "
             "examples per source, treat as directional, not definitive. ") +
            "Small dataset overall (~" + str(len(X)) + " examples); cross-"
            "validation accuracy reflects this specific data distribution, "
            "not necessarily real-world student submissions. Not a "
            "substitute for professional commercial AI detectors."
        ),
    }
    (MODEL_DIR / "ai_detector_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nСохранено в {MODEL_DIR}/")


if __name__ == "__main__":
    main()
