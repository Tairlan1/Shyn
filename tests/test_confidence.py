"""Тесты confidence_for_word_count() - честная (без жаргона) оценка
достоверности, основанная на held_out_metrics_by_scale активной модели.
Ключевое поведение, которое эти тесты защищают:
  - для multiscale-модели используются РЕАЛЬНЫЕ, по-масштабные метрики;
  - для старой (только 1500 слов) модели - честное "не измерялось",
    а не выдуманное число;
  - функция никогда не падает, даже если model_meta.json отсутствует
    или испорчен."""

from __future__ import annotations

from pathlib import Path

import app


def _reset_meta_cache(model_dir: Path):
    """confidence_for_word_count кэширует model_meta.json на модуль -
    тесты меняют app.MODEL_DIR, поэтому кэш нужно сбрасывать вручную."""
    app.MODEL_DIR = model_dir
    app._MODEL_META_CACHE = None


def test_multiscale_model_gives_real_accuracy(tmp_path):
    model_dir = tmp_path / "fake_multiscale_model"
    model_dir.mkdir()
    (model_dir / "model_meta.json").write_text(
        '{"held_out_metrics_by_scale": {'
        '"window": {"accuracy": 0.80, "macro_f1": 0.79}}}',
        encoding="utf-8",
    )
    _reset_meta_cache(model_dir)

    result = app.confidence_for_word_count(120)  # попадает в диапазон "window"
    assert result["level"] == "good"
    assert result["accuracy_pct"] == 80.0
    assert "80" not in result["text"]  # сама цифра выносится отдельным полем, не зашита в текст


def test_old_model_without_scale_metrics_is_honest_about_short_text(tmp_path):
    model_dir = tmp_path / "fake_old_model"
    model_dir.mkdir()
    (model_dir / "model_meta.json").write_text(
        '{"held_out_accuracy": 0.947}', encoding="utf-8"
    )
    _reset_meta_cache(model_dir)

    result = app.confidence_for_word_count(120)  # короткий текст, метрик по масштабам нет
    assert result["level"] == "unknown"
    assert result["accuracy_pct"] is None
    assert "не измерялась" in result["text"]


def test_old_model_trusts_long_text_it_was_actually_trained_on(tmp_path):
    model_dir = tmp_path / "fake_old_model"
    model_dir.mkdir()
    (model_dir / "model_meta.json").write_text(
        '{"held_out_accuracy": 0.947}', encoding="utf-8"
    )
    _reset_meta_cache(model_dir)

    result = app.confidence_for_word_count(1500)  # родной масштаб старой модели
    assert result["level"] == "known"
    assert result["accuracy_pct"] == 94.7


def test_missing_model_meta_does_not_crash(tmp_path):
    model_dir = tmp_path / "model_without_meta"
    model_dir.mkdir()
    _reset_meta_cache(model_dir)

    result = app.confidence_for_word_count(50)
    assert result["level"] == "unknown"
    assert result["accuracy_pct"] is None


def test_low_accuracy_scale_gives_low_confidence_wording(tmp_path):
    model_dir = tmp_path / "fake_multiscale_model"
    model_dir.mkdir()
    (model_dir / "model_meta.json").write_text(
        '{"held_out_metrics_by_scale": {'
        '"phrase": {"accuracy": 0.45, "macro_f1": 0.40}}}',
        encoding="utf-8",
    )
    _reset_meta_cache(model_dir)

    result = app.confidence_for_word_count(15)  # попадает в диапазон "phrase"
    assert result["level"] == "low"
    assert result["accuracy_pct"] == 45.0
