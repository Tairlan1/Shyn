#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
storage.py
=====================================================================
Постоянное (SQLite) хранилище сдач - drop-in замена для
`SUBMISSIONS: dict[int, dict] = {}` в app.py.

ПОЧЕМУ ЭТО БЫЛО КРИТИЧНО ИСПРАВИТЬ
-------------------------------------------------------------------
До этого изменения все сданные работы и результаты их анализа хранились
в обычном Python-словаре в оперативной памяти процесса. Это означает,
что ЛЮБОЙ перезапуск сервера (обновление кода, падение процесса,
перезагрузка машины) полностью и безвозвратно уничтожал все данные.
Для инструмента, который планируется использовать в реальном
университете, это неприемлемо - работа студента, сданная сегодня,
не должна пропадать, если сервер перезапустили завтра.

ПОЧЕМУ ИМЕННО ТАК, А НЕ ЧЕРЕЗ ORM/POSTGRES
-------------------------------------------------------------------
Каждая "сдача" - это уже готовый сложный вложенный словарь (результаты
по каждому окну текста, HTML для подсветки, вердикты и т.д.), который
собирается один раз в analyze_text() и после этого только читается
целиком. Разбивать его на нормализованные реляционные таблицы сейчас
дало бы много сложности без реальной пользы - нет ни одного места в
коде, которому нужно фильтровать/агрегировать по отдельным ПОЛЯМ внутри
сдачи (только по id целиком). Поэтому здесь сознательно простое
решение: sqlite3 (входит в стандартную библиотеку Python - без новых
зависимостей) с одной таблицей, где сдача целиком хранится как JSON.

Если проект вырастет до многопользовательской системы с реальным поиском/
фильтрацией/отчётностью по полям (например "все сдачи с ai_tier=red за
последний месяц") - это первое, что стоит заменить на нормальные таблицы
(и, вероятно, на PostgreSQL вместо SQLite для конкурентного доступа).

ИНТЕРФЕЙС
-------------------------------------------------------------------
SubmissionStore реализует минимальный набор методов словаря, которые
использует app.py (`__getitem__`, `get`, `__setitem__`, `values`,
`__len__`, `__contains__`), поэтому замена в app.py - это буквально одна
строка (см. app.py: `SUBMISSIONS = SubmissionStore(...)` вместо
`SUBMISSIONS: dict[int, dict] = {}`), без изменений во всём остальном
коде, который с ним работает.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path


class SubmissionStore:
    """Потокобезопасное (через блокировку) SQLite-хранилище сдач.

    Flask по умолчанию может обрабатывать запросы в нескольких потоках
    (см. app.run(threaded=True) или production WSGI-сервер) - без
    блокировки конкурентные записи в один sqlite-файл могли бы
    конфликтовать. Блокировка здесь простая (на весь объект, а не на
    строку) - для нагрузки уровня "группа студентов сдаёт работы" этого
    достаточно с большим запасом; для реальной высокой конкурентности
    потребовалась бы уже другая СУБД (см. докстринг модуля)."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS submissions (
                    id INTEGER PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        # check_same_thread=False: соединение создаётся заново на каждый
        # вызов (см. методы ниже), поэтому это безопасно даже без него -
        # оставлено явно для ясности, что расчёт именно на такое использование.
        return sqlite3.connect(self._db_path, check_same_thread=False)

    def __setitem__(self, sub_id: int, value: dict) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO submissions (id, data) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET data = excluded.data",
                (sub_id, payload),
            )
            conn.commit()

    def get(self, sub_id: int, default=None):
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM submissions WHERE id = ?", (sub_id,)
            ).fetchone()
        if row is None:
            return default
        return json.loads(row[0])

    def __getitem__(self, sub_id: int) -> dict:
        value = self.get(sub_id)
        if value is None:
            raise KeyError(sub_id)
        return value

    def __contains__(self, sub_id: int) -> bool:
        return self.get(sub_id) is not None

    def values(self) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT data FROM submissions ORDER BY id"
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def __len__(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM submissions").fetchone()
        return row[0]

    def next_id(self) -> int:
        """Следующий свободный id - учитывает уже сохранённые в базе сдачи
        (важно при перезапуске сервера, чтобы не выдать id, который уже
        занят существующей записью)."""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT MAX(id) FROM submissions").fetchone()
        return (row[0] or 0) + 1
