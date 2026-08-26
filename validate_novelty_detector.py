#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_novelty_detector.py
=====================================================================
Честная проверка детектора "вне распределения" (novelty, см. train_model.py
и app.NOVELTY_GATING_ENABLED) ПЕРЕД тем, как включать его в качестве
автоматического гейта поверх вердикта стилевой атрибуции.

Зачем это нужно: density-based novelty-детектор в текущей реализации
эмпирически проверялся только на маленькой, урезанной по признакам
подвыборке и показал риск ошибаться в опасную для честного студента
сторону (см. историю обсуждения) - подлинный, стилево яркий текст человека
может получать МЕНЬШИЙ novelty%, чем сглаженный, "усреднённый" ИИ-текст.
Прежде чем доверять этому сигналу в проде на полном корпусе, нужно явно
проверить направление эффекта.

Что делает скрипт:
  1. Берёт РЕАЛЬНУЮ, честно отложенную часть человеческих текстов (не
     участвовавшую в обучении модели - используйте --holdout-dir с книгами,
     не входившими в data_processed/dataset.jsonl, либо явно оставленный
     кусок книги).
  2. Берёт коллекцию известных ИИ-сгенерированных текстов (например,
     data_ai_detector/ai_written.txt, или свой набор - чем больше и
     разнообразнее по моделям/промптам, тем честнее проверка).
  3. Считает novelty% для обеих групп через ТОТ ЖЕ путь, что использует
     прод (train_model.predict_author), печатает распределения и явную
     рекомендацию: включать ли NOVELTY_GATING_ENABLED и какой порог ставить.

Запуск:
    python validate_novelty_detector.py \
        --model-dir ./model \
        --human-holdout-dir ./data_holdout \
        --ai-texts ./data_ai_detector/ai_written.txt [ещё файлы...]
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import train_model as tm


def _chunk_words(text: str, size: int = 1500, stride: int | None = None) -> list[str]:
    words = text.split()
    stride = stride or size
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - size // 2, 1), stride)]


def score_texts(texts: list[str], model_dir: Path) -> list[float]:
    scores = []
    for t in texts:
        if not t.strip():
            continue
        try:
            res = tm.predict_author(t, model_dir)
        except ValueError:
            continue
        if res["novelty_pct"] is not None:
            scores.append(res["novelty_pct"])
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=Path("./model"))
    ap.add_argument("--human-holdout-dir", type=Path, required=True,
                     help="Папка с .txt файлами ЧЕСТНО отложенных человеческих "
                          "текстов (не использованных при обучении модели).")
    ap.add_argument("--ai-texts", type=Path, nargs="+", required=True,
                     help="Один или несколько .txt файлов с известным "
                          "ИИ-сгенерированным текстом.")
    ap.add_argument("--chunk-size", type=int, default=1500)
    args = ap.parse_args()

    if not (args.model_dir / tm.OOD_FILE).exists():
        print(f"ОШИБКА: {args.model_dir / tm.OOD_FILE} не найден. "
              f"Сначала обучите модель через train_model.py train - "
              f"novelty-детектор обучается автоматически как часть train().")
        return

    human_files = sorted(args.human_holdout_dir.glob("*.txt"))
    if not human_files:
        print(f"ОШИБКА: в {args.human_holdout_dir} не найдено .txt файлов.")
        return

    human_chunks = []
    for f in human_files:
        text = f.read_text(encoding="utf-8", errors="replace")
        human_chunks.extend(_chunk_words(text, args.chunk_size))

    ai_chunks = []
    for f in args.ai_texts:
        text = f.read_text(encoding="utf-8", errors="replace")
        ai_chunks.extend(_chunk_words(text, args.chunk_size))

    print(f"Человеческих held-out фрагментов: {len(human_chunks)} (из {len(human_files)} файлов)")
    print(f"ИИ-фрагментов: {len(ai_chunks)} (из {len(args.ai_texts)} файлов)")
    if len(human_chunks) < 10 or len(ai_chunks) < 10:
        print("\nВНИМАНИЕ: меньше 10 фрагментов в одной из групп - результат "
              "статистически ненадёжен, соберите больше примеров (особенно "
              "ИИ-текстов - желательно от НЕСКОЛЬКИХ разных LLM и промптов, "
              "а не только одной модели/стиля).")

    human_scores = score_texts(human_chunks, args.model_dir)
    ai_scores = score_texts(ai_chunks, args.model_dir)

    if not human_scores or not ai_scores:
        print("ОШИБКА: не удалось получить novelty-оценки ни для одной группы.")
        return

    def summarize(name, scores):
        print(f"\n{name}: n={len(scores)}  "
              f"среднее={statistics.mean(scores):.1f}%  "
              f"медиана={statistics.median(scores):.1f}%  "
              f"мин={min(scores):.1f}%  макс={max(scores):.1f}%")

    summarize("Человеческий held-out текст", human_scores)
    summarize("ИИ-сгенерированный текст", ai_scores)

    human_mean = statistics.mean(human_scores)
    ai_mean = statistics.mean(ai_scores)
    # Простая, консервативная метрика разделения: доля человеческих
    # фрагментов, чей novelty% выше медианы ИИ-фрагментов (AUC-подобная
    # прикидка без лишних зависимостей).
    ai_median = statistics.median(ai_scores)
    separation = sum(1 for s in human_scores if s > ai_median) / len(human_scores)

    print(f"\nДоля человеческих фрагментов с novelty% выше медианы ИИ-группы: "
          f"{separation*100:.1f}% (в идеале - близко к 100%)")

    print("\n" + "=" * 70)
    if human_mean > ai_mean and separation >= 0.75:
        suggested_threshold = round((statistics.median(human_scores) + ai_median) / 2, 1)
        print("РЕКОМЕНДАЦИЯ: сигнал работает в ожидаемую сторону (человеческий "
              "текст статистически типичнее ИИ-текста для этой модели).")
        print(f"Можно включить app.NOVELTY_GATING_ENABLED = True и выставить "
              f"app.NOVELTY_UNRELIABLE_MAX ≈ {suggested_threshold}% как отправную "
              f"точку - но перепроверьте на большем и разнообразном наборе ИИ-"
              f"текстов (разные LLM, разные промпты, разная длина) перед продом.")
    else:
        print("РЕКОМЕНДАЦИЯ: НЕ включать app.NOVELTY_GATING_ENABLED.")
        print("Сигнал либо не разделяет группы, либо работает в обратную сторону "
              "(ИИ-текст выглядит типичнее человеческого для этой модели) - "
              "включение гейтинга в этом состоянии рискует несправедливо "
              "занижать вердикт подлинным работам студентов. Полагайтесь на "
              "AI Detection (ai_tier) как основной автоматический сигнал; "
              "novelty% оставьте только как информационную метрику на странице "
              "отчёта, без влияния на вердикт.")
    print("=" * 70)


if __name__ == "__main__":
    main()
