#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_heuristics.py
=====================================================================
ВАЖНО: в проекте нет обученного детектора ИИ-текста (модель в model/
решает только задачу атрибуции авторства среди 5 писателей). Надёжных,
научно валидированных детекторов ИИ-текста в принципе не существует —
любые методы (в т.ч. коммерческие) дают заметный процент ложных
срабатываний.

История версии (для прозрачности): первая версия этой эвристики
пробовала добавить статистическую языковую модель (word-level trigram)
для сигнала "перплексии" - идея, на которой построены настоящие
AI-детекторы вроде GPTZero. Она была реализована и протестирована
дважды: сначала на корпусе пяти авторов проекта (оказалось, что модель
просто помечает как подозрительное любое современное слово - то есть
дублирует лексикон ниже, а не даёт независимый сигнал), затем на
жанрово-нейтральном корпусе (nltk brown+reuters, ~2.9 млн слов) - но
триграммная модель на корпусе такого размера оказалась слишком
разреженной: она реагирует в основном на несовпадение домена/жанра
(единичная фраза, которой не было в конкретных ~3 млн слов обучения),
а не на "ИИ-подобность" - на тесте синтетический ИИ-абзац получил ДАЖЕ
БОЛЕЕ высокую "неожиданность", чем подлинная викторианская проза, то
есть сигнал был не просто слабым, а вводящим в заблуждение. От этого
подхода отказались как от ненадёжного при доступных объёмах данных, а
не встроили с оговоркой - ложная строгость хуже отсутствия сигнала.

Вместо этого эвристика опирается на четыре сигнала, которые лучше
задокументированы для распознавания текста, стилизованного под работу
LLM-ассистента, и не требуют статистической языковой модели:

  1. Лексикон "ИИ-маркеров" — слова и обороты, статистически
     перегруженные в текстах современных LLM ("GPT-измы": delve,
     tapestry, boundaries, moreover, seamless, robust, leverage,
     navigate, landscape, "it's important to note" и т.п.), взвешенные
     по характерности (частые обороты вроде "moreover" сами по себе
     слабый сигнал, редкие клише вроде "delve into the tapestry" -
     намного более специфичный).
  2. Плотность и кластеризация дискурсивных маркеров — доля
     предложений/абзацев, начинающихся с переходных оборотов
     (Moreover, Furthermore, Additionally, In conclusion...). У живого
     текста переходы используются нечасто и не в каждом абзаце; у
     сгенерированного текста они часто идут почти в каждом предложении
     подряд - это сильный, легко объяснимый сигнал.
  3. Паттерн "перечисление из трёх" ("speed, accuracy, and
     reliability") - задокументированная особенность LLM-текста,
     повторяющаяся заметно чаще, чем в среднем у человека.
  4. Однородность длины предложений — человеческая проза пишет
     "рвано"; подозрительно ровный ритm предложений (низкий
     коэффициент вариации) статистически чаще встречается в
     ИИ-подобном тексте.

Результат — число 0..100, которое стоит показывать пользователю ИМЕННО
как "эвристический индикатор", а не как доказанный процент вероятности.
"""

from __future__ import annotations

import re

import numpy as np

import preprocess_corpus as prep

# ---------------------------------------------------------------------------
# 1. Лексикон. Разбит на два веса: "сильные" клише (редко встречаются
#    в естественном тексте вне LLM-генераций) и "слабые" переходные
#    слова (сами по себе обычные, подозрительны только в высокой
#    концентрации - их основной вклад идёт через сигнал №2).
# ---------------------------------------------------------------------------
STRONG_MARKERS = [
    "delve", "delving", "delves", "tapestry", "boundaries", "underscore",
    "underscores", "underscoring", "seamless", "seamlessly", "leverage",
    "leveraging", "multifaceted", "intricacies", "holistic", "holistically",
    "paradigm shift", "synergy", "synergistic", "cutting-edge",
    "state-of-the-art", "ever-evolving", "ever-changing",
    "it is important to note", "it's important to note",
    "it is worth noting", "it's worth noting", "plays a crucial role",
    "plays a significant role", "pivotal role", "a testament to",
    "is a testament", "unwavering commitment", "unwavering dedication",
    "invaluable", "comprehensive understanding", "garnered", "garner",
    "bolster", "bolstering", "embark on a journey", "resonates deeply",
    "myriad of", "plethora of", "in the realm of", "at the heart of",
    "when it comes to", "navigate the complexities", "navigate the",
    "as an ai", "as a language model", "i cannot", "i don't have personal",
    "i don't have the ability", "foster a deeper understanding",
    "rich tapestry", "vibrant tapestry",
]

WEAK_TRANSITIONS = [
    "moreover", "furthermore", "additionally", "in conclusion",
    "notably", "arguably", "in essence", "in summary", "to summarize",
    "overall,", "consequently", "therefore,", "thus,", "indeed,",
    "however,", "nonetheless,", "nevertheless,", "in addition,",
    "on the other hand,", "as a result,",
]

MODERN_ANACHRONISMS = [
    "internet", "online", "email", "smartphone", "computer", "website",
    "app", "software", "algorithm", "data-driven",
]

_STRONG_RE = re.compile(
    r"(?<![a-zA-Z])(" + "|".join(re.escape(p) for p in STRONG_MARKERS) + r")(?![a-zA-Z])",
    re.IGNORECASE,
)
_WEAK_RE = re.compile(
    r"(?<![a-zA-Z])(" + "|".join(re.escape(p) for p in WEAK_TRANSITIONS) + r")",
    re.IGNORECASE,
)
_ANACHRONISM_RE = re.compile(
    r"(?<![a-zA-Z])(" + "|".join(re.escape(p) for p in MODERN_ANACHRONISMS) + r")(?![a-zA-Z])",
    re.IGNORECASE,
)

# Оборот в начале предложения, сигнализирующий "дискурсивный маркер"
# (для сигнала №2 - плотность/кластеризация переходов).
_SENTENCE_OPENER_RE = re.compile(
    r"^\s*(" + "|".join(re.escape(p.rstrip(",")) for p in WEAK_TRANSITIONS) +
    r"|In today's|In this|This comprehensive|This essay|This analysis)\b",
    re.IGNORECASE,
)

# "X, Y, and Z" / "X, Y, or Z" - перечисление из трёх однородных членов.
_LIST_OF_THREE_RE = re.compile(
    r"\b\w+(?:,\s+\w[\w'-]*){1,2},?\s+(?:and|or)\s+\w[\w'-]*\b"
)


def _lexicon_signal(text: str, n_words: int) -> tuple[float, list[str]]:
    strong_hits = _STRONG_RE.findall(text)
    weak_hits = _WEAK_RE.findall(text)
    anachronism_hits = _ANACHRONISM_RE.findall(text)

    # сильные маркеры весят в 3 раза больше слабых переходов
    score = 3.0 * len(strong_hits) + 1.0 * len(weak_hits) + 2.0 * len(anachronism_hits)
    per_1000 = 1000.0 * score / max(1, n_words)
    signal = float(np.clip(per_1000 / 10.0, 0.0, 1.0))
    all_hits = strong_hits + weak_hits + anachronism_hits
    return signal, all_hits


def _discourse_density_signal(sentences: list[str]) -> float:
    if len(sentences) < 3:
        return 0.0
    openers = sum(1 for s in sentences if _SENTENCE_OPENER_RE.match(s.strip()))
    ratio = openers / len(sentences)
    # >=35% предложений начинаются с явного дискурсивного маркера -
    # уже сильно нетипично для человеческого текста такого объёма.
    return float(np.clip(ratio / 0.35, 0.0, 1.0))


def _list_of_three_signal(text: str, n_words: int) -> float:
    hits = len(_LIST_OF_THREE_RE.findall(text))
    per_500 = 500.0 * hits / max(1, n_words)
    # 2+ таких перечисления на 500 слов - уже заметно выше типичной
    # частоты в естественной прозе.
    return float(np.clip(per_500 / 2.0, 0.0, 1.0))


def _sentence_uniformity_signal(sentences: list[str]) -> float:
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 3:
        return 0.0
    mean = float(np.mean(lengths))
    std = float(np.std(lengths))
    if mean <= 0:
        return 0.0
    cv = std / mean
    # человеческая проза обычно cv в районе 0.5-0.9; заметно более
    # ровный ритм (cv < 0.35) - более "ИИ-подобный" паттерн.
    return float(np.clip((0.55 - cv) / 0.4, 0.0, 1.0))


def score_fragment(text: str) -> dict:
    """Возвращает эвристический AI-индикатор для одного фрагмента текста.

    ВНИМАНИЕ: это НЕ обученный классификатор и не валидированный
    детектор — см. докстринг модуля. Использовать только как
    ориентировочную, а не доказательную оценку.
    """
    words = [w for w in text.split() if any(c.isalpha() for c in w)]
    n_words = max(1, len(words))
    sentences = prep.split_sentences(text)

    lex_signal, marker_hits = _lexicon_signal(text, n_words)
    discourse_signal = _discourse_density_signal(sentences)
    list3_signal = _list_of_three_signal(text, n_words)
    uniformity_signal = _sentence_uniformity_signal(sentences)

    score = (
        0.40 * lex_signal
        + 0.30 * discourse_signal
        + 0.15 * list3_signal
        + 0.15 * uniformity_signal
    )
    return {
        "ai_score": float(np.clip(score, 0.0, 1.0)),
        "lexicon_signal": lex_signal,
        "discourse_signal": discourse_signal,
        "list_of_three_signal": list3_signal,
        "uniformity_signal": uniformity_signal,
        "marker_hits": marker_hits,
    }
