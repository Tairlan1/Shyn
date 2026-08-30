"""Общие фикстуры pytest. Добавляет корень репозитория в sys.path, чтобы
тесты могли делать `import app`, `import storage` и т.д. без установки
проекта как пакета."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
