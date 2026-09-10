#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auth.py
=====================================================================
Минимальная, но настоящая авторизация для app.py.

ПОЧЕМУ ЭТО ПОТРЕБОВАЛОСЬ
-------------------------------------------------------------------
До этого изменения у app.py вообще не было понятия пользователя: шапка
(base.html) показывала жёстко зашитое имя "Тайлан С. · гр. ПИ-21",
`/dashboard` показывал ВСЕ сданные работы всех студентов в одной таблице,
а `/report/<sub_id>` и `/report/<sub_id>/simple` открывали разбор любой
сдачи любому, кто знает (или перебирает) числовой id — id выдаются
последовательно (`SubmissionStore.next_id`), так что подобрать чужой id
тривиально. Тем самым содержимое чужой работы и вердикт по ней (включая
AI-детект) были доступны кому угодно без какой-либо проверки личности.

ЧТО ИСПРАВЛЕНО
-------------------------------------------------------------------
  - Появилась настоящая (хоть и простая — без БД пользователей, для
    учебного/демо-масштаба этого достаточно) сессионная авторизация:
    логин/пароль -> подписанная Flask-сессия (см. app.py: SECRET_KEY).
  - Каждая сдача теперь помечается owner_login при создании
    (см. app.py::make_submission).
  - `/dashboard` показывает студенту только ЕГО собственные работы;
    роль teacher видит все (нужно преподавателю для проверки).
  - `/report/<id>` и `/report/<id>/simple` проверяют owner_login == текущий
    пользователь ИЛИ роль teacher — иначе 403, а не тихо отдают чужой отчёт.
  - Все роуты, кроме /login и статики, требуют аутентификации
    (см. app.py::require_login, зарегистрирован через before_request).

Учётные записи ниже — DEMO_ACCOUNTS, захардкоженные для демо/учебного
стенда (как и STUDENT_ACCOUNTS в UniPlatform) — в реальном вузовском
развёртывании этот модуль стоит заменить на интеграцию с реальным
IdP/LDAP/SSO вуза, сохранив тот же контракт (current_user() -> dict с
login/role, ownership-проверки в app.py остаются без изменений).
"""

from __future__ import annotations

from functools import wraps

from flask import abort, session

# login -> {password, role, display, group}
# role: "student" | "teacher". Пароли — открытым текстом, т.к. это учебный
# демо-стенд без реальных персональных данных; для боевого использования
# замените на реальный IdP (см. докстринг модуля выше) вместо хранения
# паролей в коде вообще.
DEMO_ACCOUNTS: dict[str, dict] = {
    "a.serikova": {"password": "student2026", "role": "student",
                   "display": "Серикова Айгерим", "group": "ПИ-21"},
    "n.bekov": {"password": "student2026", "role": "student",
                "display": "Беков Нурлан", "group": "ПИ-21"},
    "d.tanirbergen": {"password": "student2026", "role": "student",
                       "display": "Танирберген Дана", "group": "ПИ-22"},
    "teacher": {"password": "teacher2026", "role": "teacher",
                "display": "Ким Руслан Сергеевич", "group": "Кафедра ИС"},
}

# Владелец демо-сдач, которые seed_demo_submissions() создаёт при первом
# запуске на пустом хранилище (см. app.py) — чтобы у них тоже был владелец
# и они не "утекали" всем подряд студентам через dashboard/report.
DEMO_SEED_OWNER = "a.serikova"


def authenticate(login: str, password: str) -> dict | None:
    """Проверяет логин/пароль. Возвращает публичный профиль (без пароля)
    или None, если данные неверны."""
    account = DEMO_ACCOUNTS.get(login)
    if account is None or account["password"] != password:
        return None
    return {"login": login, "role": account["role"],
            "display": account["display"], "group": account["group"]}


def login_user(profile: dict) -> None:
    session.clear()
    session["login"] = profile["login"]
    session["role"] = profile["role"]
    session["display"] = profile["display"]
    session["group"] = profile["group"]
    session.permanent = True


def logout_user() -> None:
    session.clear()


def current_user() -> dict | None:
    """Профиль вошедшего пользователя из подписанной сессии, либо None."""
    login = session.get("login")
    if not login:
        return None
    return {
        "login": login,
        "role": session.get("role", "student"),
        "display": session.get("display", login),
        "group": session.get("group", ""),
    }


def is_teacher() -> bool:
    user = current_user()
    return bool(user and user["role"] == "teacher")


def can_view_submission(sub: dict) -> bool:
    """Владелец сдачи или преподаватель — иначе нет доступа к отчёту."""
    user = current_user()
    if user is None:
        return False
    if user["role"] == "teacher":
        return True
    return sub.get("owner_login") == user["login"]


def require_submission_access(sub: dict) -> None:
    """abort(403), если у текущего пользователя нет доступа к этой сдаче.
    Используется вместо тихого сокрытия (404), чтобы явно сообщать о
    нехватке прав, а не притворяться, что сдачи не существует - при
    404 пользователь мог бы решить, что просто ошибся ссылкой/id и
    продолжить перебор; 403 однозначно говорит "это не ваш отчёт"."""
    if not can_view_submission(sub):
        abort(403)


def login_required(view):
    """Декоратор для роутов, которым явно неудобно проверяться через общий
    before_request в app.py (например, если понадобится точечно освободить
    какой-то путь). Основная защита всё же - require_login() в app.py,
    этот декоратор - дополнительный явный слой на конкретных роутах."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_user() is None:
            abort(401)
        return view(*args, **kwargs)
    return wrapped
