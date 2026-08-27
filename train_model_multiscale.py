#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_model_multiscale.py
=====================================================================
Дообучение модели авторского стиля на ФРАГМЕНТАХ РАЗНОГО РАЗМЕРА, а не
только на ~1500-словных чанках, на которых обучен текущий
model/author_style_pipeline.joblib.

ПРОБЛЕМА, которую решает этот скрипт
-------------------------------------------------------------------
Модель обучена ИСКЛЮЧИТЕЛЬНО на больших (~1500 слов) фрагментах
(data_processed/dataset.jsonl, см. preprocess_corpus.py). Она никогда не
видела короткие фрагменты во время обучения. Поэтому когда приложение
просит её оценить стиль по маленькому окну подсветки (~180 слов) или
тем более по одному предложению/словосочетанию, она оказывается вне
своего обучающего распределения и даёт шумные, малоточные оценки. Это
классическое несоответствие длины train/inference (train-test length
mismatch), а не баг в коде приложения - его нельзя починить настройкой
порогов, только переобучением на данных того же масштаба, что и реальный
инференс.

РЕШЕНИЕ
-------------------------------------------------------------------
Взять уже существующий, размеченный по авторам датасет
(data_processed/dataset.jsonl) и дополнить его версиями ТЕХ ЖЕ текстов,
нарезанными на несколько более мелких масштабов:

    large       ~1500 слов   (оригинальные чанки, как сейчас)
    medium       600-900 слов
    semi_small   250-400 слов
    small         80-150 слов
    phrase         10-25 слов  (короткие словосочетания)

Все производные фрагменты - это подстроки РЕАЛЬНОГО, уже вычитанного
текста конкретного автора (не синтетика и не перефразировка), поэтому
разметка (author) остаётся достоверной по построению - мы просто режем
уже правильно размеченный текст на куски поменьше.

Модель, обученная на всех масштабах одновременно, учится распознавать
авторский стиль независимо от длины входного текста - именно это нужно
для точной посегментной/пословной подсветки.

ЧЕСТНОСТЬ ОЦЕНКИ - ПОЧЕМУ ЭТО НЕ УТЕЧКА ДАННЫХ
-------------------------------------------------------------------
Группировка для train/test-разбиения и для GroupKFold внутри
train_model.train() идёт ПО КНИГЕ (author::book), а не по отдельному
фрагменту. Мелкие производные фрагменты одного и того же чанка/книги
всегда попадают в ту же группу (train ИЛИ test целиком), что и их
"родитель" - иначе один и тот же исходный текст в разной нарезке мог бы
одновременно оказаться и в train, и в test, что завысило бы метрики
обманчиво. Этот скрипт сохраняет исходное поле "book" у каждого
производного фрагмента ровно поэтому.

КАК ЭТО СООТНОСИТСЯ С train_model.py
-------------------------------------------------------------------
Этот скрипт НЕ дублирует обучение классификатора - он только строит
расширенный датасет (data_processed/dataset_multiscale.jsonl) и затем
вызывает уже существующую, проверенную функцию train_model.train() на
этом новом файле. Используется тот же пайплайн (TF-IDF + стилометрия,
StackingClassifier, честная out-of-fold калибровка novelty-детектора)
- меняются только данные, а не методология.

Запуск
-------------------------------------------------------------------
    python3 train_model_multiscale.py

Что произойдёт:
  1. Прочитает data_processed/dataset.jsonl (уже существующий у вас датасет).
  2. Сгенерирует дополнительные короткие/средние фрагменты для каждого
     чанка (см. SCALES ниже) и сохранит объединённый датасет в
     data_processed/dataset_multiscale.jsonl (обычный jsonl, можно
     посмотреть глазами / прогнать через любой текстовый редактор).
  3. Запустит train_model.train() на этом датасете. Результат сохранится
     в ./model_multiscale/ - папка ./model НЕ трогается, чтобы можно
     было сравнить старую и новую модель бок о бок перед переключением.

После обучения, если результат устраивает:
  - app.py:          python app.py --model-dir model_multiscale
  - api_analyze.py:  поменять MODEL_DIR на Path("model_multiscale")
                      (или переименовать папки: model -> model_1500only,
                      model_multiscale -> model)

Необязательные флаги:
    --data PATH        путь к исходному dataset.jsonl (по умолчанию
                        data_processed/dataset.jsonl)
    --output-dataset PATH   куда сохранить multiscale-датасет
    --model-dir PATH   куда сохранить обученную модель
    --skip-train       только построить датасет, не обучать модель сразу
                        (удобно, если хотите сначала глазами проверить
                        data_processed/dataset_multiscale.jsonl)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import preprocess_corpus as prep
import train_model as tm

PROJECT_DIR = Path(__file__).parent
DEFAULT_SOURCE = PROJECT_DIR / "data_processed" / "dataset.jsonl"
DEFAULT_OUTPUT_DATASET = PROJECT_DIR / "data_processed" / "dataset_multiscale.jsonl"
DEFAULT_MODEL_DIR = PROJECT_DIR / "model_multiscale"

RANDOM_SEED = 42

# (имя_масштаба, мин_слов, макс_слов, макс_фрагментов_на_один_исходный_чанк)
# Ограничение "макс_фрагментов" не даёт мелким масштабам задавить датасет
# числом почти дублирующих друг друга фрагментов из одной и той же книги -
# без него, например, "phrase" дал бы в разы больше примеров, чем "large",
# и модель могла бы начать игнорировать крупный масштаб при обучении.
SCALES = [
    ("medium", 600, 900, 2),
    ("semi_small", 250, 400, 3),
    ("small", 80, 150, 3),
    ("phrase", 10, 25, 4),
]


def _word_count(text: str) -> int:
    return len(text.split())


def _sentence_windows(sentences: list[str], lo: int, hi: int,
                       max_windows: int, rng: random.Random) -> list[str]:
    """Копит предложения подряд, пока суммарная длина не попадёт в [lo, hi],
    затем начинает копить заново с нуля (не строго непересекающиеся окна -
    нам нужно разнообразие длины и содержания, а не идеальное покрытие без
    повторов). Если кандидатов получилось больше max_windows - берётся
    воспроизводимая случайная подвыборка."""
    candidates = []
    buf: list[str] = []
    buf_words = 0
    for sentence in sentences:
        buf.append(sentence)
        buf_words += _word_count(sentence)
        if buf_words >= lo:
            if buf_words <= hi:
                candidates.append(" ".join(buf))
            buf = []
            buf_words = 0
    if not candidates:
        return []
    if len(candidates) <= max_windows:
        return candidates
    return rng.sample(candidates, max_windows)


def _phrase_windows(words: list[str], lo: int, hi: int,
                     max_windows: int, rng: random.Random) -> list[str]:
    """Для масштаба 'phrase' предложения почти всегда длиннее верхней
    границы целиком, поэтому здесь режем не по границам предложений, а
    чисто по словесным окнам фиксированного размера - это и есть
    "словосочетания", а не обязательно грамматически полные фразы."""
    if len(words) < lo:
        return []
    size = min(hi, max(lo, len(words)))
    candidates = []
    i = 0
    while i + lo <= len(words):
        end = min(i + size, len(words))
        candidates.append(" ".join(words[i:end]))
        i += size
    if len(candidates) <= max_windows:
        return candidates
    return rng.sample(candidates, max_windows)


def build_multiscale_dataset(source_path: Path, output_path: Path) -> None:
    if not source_path.exists():
        raise SystemExit(
            f"Не найден {source_path} - сначала запустите preprocess_corpus.py, "
            f"как описано в README проекта, чтобы получить исходный dataset.jsonl."
        )

    rng = random.Random(RANDOM_SEED)
    out_records = []
    n_source = 0

    with source_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_source += 1
            text = rec["text"]
            author = rec["author"]
            book = rec["book"]
            chunk_id = rec["chunk_id"]

            # Оригинальный большой чанк - без изменений, помечен scale="large".
            out_records.append({
                "author": author, "book": book, "chunk_id": chunk_id,
                "text": text, "word_count": rec.get("word_count", _word_count(text)),
                "scale": "large",
            })

            sentences = prep.split_sentences(text)
            words = text.split()

            for scale_name, lo, hi, max_windows in SCALES:
                if scale_name == "phrase":
                    frags = _phrase_windows(words, lo, hi, max_windows, rng)
                else:
                    frags = _sentence_windows(sentences, lo, hi, max_windows, rng)
                for j, frag_text in enumerate(frags):
                    out_records.append({
                        "author": author, "book": book,
                        "chunk_id": f"{chunk_id}_{scale_name}_{j}",
                        "text": frag_text, "word_count": _word_count(frag_text),
                        "scale": scale_name,
                    })

    rng.shuffle(out_records)  # порядок в файле не должен коррелировать со scale/книгой

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_scale: dict[str, int] = {}
    for rec in out_records:
        by_scale[rec["scale"]] = by_scale.get(rec["scale"], 0) + 1

    print(f"Исходных чанков (scale=large): {n_source}")
    print(f"Итоговый multiscale-датасет: {len(out_records)} фрагментов -> {output_path}")
    for scale in ("large", "medium", "semi_small", "small", "phrase"):
        if scale in by_scale:
            print(f"  {scale:12s}: {by_scale[scale]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--output-dataset", type=Path, default=DEFAULT_OUTPUT_DATASET)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--skip-train", action="store_true",
                     help="только построить датасет, не запускать обучение сразу")
    args = ap.parse_args()

    print("Шаг 1/2: строю multiscale-датасет из существующего "
          f"{args.data} ...")
    build_multiscale_dataset(args.data, args.output_dataset)

    if args.skip_train:
        print("\n--skip-train указан - обучение пропущено. Датасет сохранён, "
              "можно посмотреть глазами и запустить обучение отдельно:")
        print(f"    python3 train_model.py train --data {args.output_dataset} "
              f"--model-dir {args.model_dir}")
        return

    print("\nШаг 2/2: обучаю модель на multiscale-датасете "
          "(переиспользую train_model.train, та же методология)...")
    print(f"Результат будет сохранён в {args.model_dir} "
          f"(папка ./model не трогается - можно сравнить старую и новую модель).")
    tm.train(args.output_dataset, args.model_dir,
              cv_folds=args.cv_folds, test_size=args.test_size,
              random_state=RANDOM_SEED)

    print("\nГотово. Чтобы переключить платформу на новую модель:")
    print(f"  app.py:          python app.py --model-dir {args.model_dir}")
    print(f"  api_analyze.py:  поменяйте MODEL_DIR на Path('{args.model_dir.name}') "
          f"или переименуйте папки (model -> model_1500only, "
          f"{args.model_dir.name} -> model).")
    print("\nПеред переключением в проде сравните held-out accuracy/F1 старой и "
          "новой модели (см. вывод обучения выше) и, если добавляли ai_detector "
          "с несколькими источниками ранее, проверьте, что подсветка на реальных "
          "коротких фрагментах субъективно стала точнее, а не просто иначе шумной.")


if __name__ == "__main__":
    main()
