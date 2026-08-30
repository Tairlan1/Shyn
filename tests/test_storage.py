"""Тесты storage.SubmissionStore - в первую очередь то, ради чего это
хранилище вообще писалось: данные должны переживать создание НОВОГО
объекта SubmissionStore (что эквивалентно перезапуску процесса Flask,
т.к. старый объект и его состояние в памяти исчезают)."""

from __future__ import annotations

from pathlib import Path

import pytest

from storage import SubmissionStore


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


def test_set_and_get(db_path):
    store = SubmissionStore(db_path)
    store[1] = {"title": "Тестовая работа", "n_words": 42}
    assert store[1] == {"title": "Тестовая работа", "n_words": 42}


def test_get_missing_returns_default(db_path):
    store = SubmissionStore(db_path)
    assert store.get(999) is None
    assert store.get(999, "запасное значение") == "запасное значение"


def test_getitem_missing_raises_keyerror(db_path):
    store = SubmissionStore(db_path)
    with pytest.raises(KeyError):
        store[999]


def test_contains(db_path):
    store = SubmissionStore(db_path)
    assert 1 not in store
    store[1] = {"title": "x"}
    assert 1 in store


def test_len_and_values(db_path):
    store = SubmissionStore(db_path)
    assert len(store) == 0
    store[1] = {"title": "первая"}
    store[2] = {"title": "вторая"}
    assert len(store) == 2
    assert {v["title"] for v in store.values()} == {"первая", "вторая"}


def test_next_id_increments(db_path):
    store = SubmissionStore(db_path)
    assert store.next_id() == 1
    store[1] = {"title": "x"}
    assert store.next_id() == 2
    store[5] = {"title": "y"}  # пропуск id (например, после ручного удаления)
    assert store.next_id() == 6


def test_update_existing_id_overwrites(db_path):
    store = SubmissionStore(db_path)
    store[1] = {"title": "версия 1"}
    store[1] = {"title": "версия 2"}
    assert len(store) == 1
    assert store[1]["title"] == "версия 2"


def test_data_survives_new_store_instance(db_path):
    """Главный сценарий, ради которого писался этот класс: новый объект
    SubmissionStore (= новый процесс Flask после перезапуска) должен
    видеть данные, записанные СТАРЫМ объектом."""
    store1 = SubmissionStore(db_path)
    store1[1] = {"title": "Работа, сданная до перезапуска сервера"}
    del store1

    store2 = SubmissionStore(db_path)  # "новый процесс"
    assert 1 in store2
    assert store2[1]["title"] == "Работа, сданная до перезапуска сервера"
    assert store2.next_id() == 2  # не начинает нумерацию заново с 1
