#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_models.py
=====================================================================
Сравнивает СТАРУЮ модель (./model - обучена только на ~1500-словных
чанках) и НОВУЮ (./model_multiscale - обучена на смеси масштабов) на
ОДНИХ И ТЕХ ЖЕ фрагментах текста, в разбивке по масштабу (large / medium
/ semi_small / window / sentence / phrase).

ЗАЧЕМ ЭТО НУЖНО, А НЕ ПРОСТО СМОТРЕТЬ НА МЕТРИКИ ИЗ ЛОГА ОБУЧЕНИЯ
-------------------------------------------------------------------
model_meta.json у каждой модели содержит held-out accuracy, но это
метрики С РАЗНЫХ прогонов train() - разные случайные тестовые сплиты
(разные отложенные книги), и напрямую сравнивать "94.7% у старой" с
"72.6% у новой" некорректно: они посчитаны не на одних и тех же
примерах. Этот скрипт устраняет это - обе модели прогоняются на ОДНОМ
и том же наборе фрагментов.

ВАЖНАЯ ОГОВОРКА ПО ЧЕСТНОСТИ (прочитайте перед тем как доверять цифрам)
-------------------------------------------------------------------
И старая, и новая модель в train_model.py в конце ДООБУЧАЮТСЯ на 100%
своих данных перед сохранением (см. "Дообучение финальной модели на
100% данных" в логе обучения) - то есть по-настоящему "невиданных"
данных для уже сохранённых моделей не существует в принципе, и этот
скрипт использует ФРАГМЕНТЫ ИЗ dataset_multiscale.jsonl, то есть какая-то
часть из них технически участвовала в обучении ОБЕИХ моделей (или
только одной из них - см. ниже). Что это означает по масштабам:

  - Для scale=large: обе модели видели такие фрагменты при обучении
    (это и есть их родной train-набор) - сравнение здесь НЕ доказывает
    обобщение, а скорее показывает, не просела ли новая модель на своей
    базовой задаче после добавления мелких масштабов.
  - Для scale != large (medium/semi_small/window/sentence/phrase):
    СТАРАЯ модель никогда не видела фрагментов такого масштаба вообще
    (их не существовало на момент её обучения) - для неё это честный,
    непредвзятый тест "как старая модель справляется с коротким
    текстом". Именно эти строки таблицы отвечают на исходный вопрос.

Короче: смотрите в первую очередь на строки, ГДЕ scale != large - там
сравнение содержательно и честно в пользу вопроса "стало ли лучше".

Запуск
-------------------------------------------------------------------
    python3 compare_models.py
    python3 compare_models.py --old-model-dir model --new-model-dir model_multiscale
    python3 compare_models.py --n-per-scale 300   # больше примеров на масштаб (медленнее)
"""

from __future__ import annotations

import argparse
import __main__
import json
import random
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", category=UserWarning)

import train_model as tm

# Чинит распаковку моделей, обученных запуском "python train_model.py ..."
# напрямую (в этом случае кастомные классы вроде StylometricFeaturizer
# пиклятся под модулем __main__, а не train_model).
__main__.StylometricFeaturizer = tm.StylometricFeaturizer

PROJECT_DIR = Path(__file__).parent
DATASET_PATH = PROJECT_DIR / "data_processed" / "dataset_multiscale.jsonl"

SCALE_ORDER = ["large", "medium", "semi_small", "window", "sentence", "phrase"]


def load_multiscale(path: Path):
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
    return records


def sample_per_scale(records, n_per_scale: int, seed: int):
    rng = random.Random(seed)
    by_scale: dict[str, list] = {}
    for rec in records:
        by_scale.setdefault(rec.get("scale", "large"), []).append(rec)

    sampled = {}
    for scale, recs in by_scale.items():
        if len(recs) <= n_per_scale:
            sampled[scale] = recs
        else:
            sampled[scale] = rng.sample(recs, n_per_scale)
    return sampled


def evaluate(pipeline, le, records) -> tuple[float, float]:
    """Возвращает (accuracy, среднее_P(истинный_автор))."""
    texts = [r["text"] for r in records]
    true_authors = [r["author"] for r in records]

    proba = pipeline.predict_proba(texts)
    classes = list(le.classes_)

    correct = 0
    true_probs = []
    for i, true_author in enumerate(true_authors):
        pred_idx = int(np.argmax(proba[i]))
        if classes[pred_idx] == true_author:
            correct += 1
        if true_author in classes:
            true_probs.append(float(proba[i][classes.index(true_author)]))

    n = len(records)
    acc = correct / n if n else 0.0
    mean_true_p = float(np.mean(true_probs)) if true_probs else 0.0
    return acc, mean_true_p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DATASET_PATH)
    ap.add_argument("--old-model-dir", type=Path, default=Path("model"))
    ap.add_argument("--new-model-dir", type=Path, default=Path("model_multiscale"))
    ap.add_argument("--n-per-scale", type=int, default=200,
                     help="сколько случайных фрагментов на масштаб брать для сравнения")
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()

    if not args.dataset.exists():
        print(f"ОШИБКА: {args.dataset} не найден. Сначала запустите:\n"
              f"  python train_model.py build-multiscale --data data_processed/dataset.jsonl "
              f"--output {args.dataset}")
        sys.exit(1)

    print(f"Загружаю {args.old_model_dir} (старая, только 1500 слов) ...")
    old_pipeline, old_le = tm.load_model(args.old_model_dir)
    print(f"Загружаю {args.new_model_dir} (новая, multiscale) ...")
    new_pipeline, new_le = tm.load_model(args.new_model_dir)

    print(f"Загружаю {args.dataset} и беру до {args.n_per_scale} случайных "
          f"фрагментов на масштаб (seed={args.seed}) ...")
    records = load_multiscale(args.dataset)
    sampled = sample_per_scale(records, args.n_per_scale, args.seed)

    print()
    header = f"{'scale':12s} {'n':>6s}   {'СТАРАЯ acc':>11s} {'СТАРАЯ P(true)':>15s}   " \
             f"{'НОВАЯ acc':>10s} {'НОВАЯ P(true)':>14s}   {'Δ acc':>7s}"
    print(header)
    print("-" * len(header))

    for scale in SCALE_ORDER:
        if scale not in sampled:
            continue
        recs = sampled[scale]
        old_acc, old_p = evaluate(old_pipeline, old_le, recs)
        new_acc, new_p = evaluate(new_pipeline, new_le, recs)
        delta = new_acc - old_acc
        flag = "  <- честное сравнение (старая такого не видела)" if scale != "large" else "  (обе видели при обучении)"
        print(f"{scale:12s} {len(recs):6d}   "
              f"{old_acc*100:10.1f}% {old_p*100:14.1f}%   "
              f"{new_acc*100:9.1f}% {new_p*100:13.1f}%   "
              f"{delta*100:+6.1f}%{flag}")

    print()
    print("Как читать:")
    print("  - 'acc' - доля фрагментов, где топ-1 предсказание = истинный автор.")
    print("  - 'P(true)' - средняя вероятность, которую модель приписала ИМЕННО")
    print("    истинному автору (даже если он не оказался топ-1) - полезнее для")
    print("    highlighting, т.к. именно это число показывается как % совпадения.")
    print("  - Строки со scale != large - главный ответ на вопрос 'стало ли лучше")
    print("    на коротких фрагментах' (см. оговорку в докстринге файла).")


if __name__ == "__main__":
    main()
