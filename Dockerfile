# Единый образ для обоих входов (app.py - веб-интерфейс, api_analyze.py -
# REST API для UniPlatform) - они используют один и тот же код и модели,
# отдельные образы только дублировали бы зависимости. Какой процесс
# запускать, определяет docker-compose.yml (или явная команда при `docker run`).

FROM python:3.12-slim

WORKDIR /app

# Системные зависимости для pdfplumber (извлечение текста из PDF) и сборки
# некоторых wheels scikit-learn/scipy на архитектурах без готовых бинарников.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NLTK punkt - используется preprocess_corpus.py/split_sentences с graceful
# fallback на regex при отсутствии, но с ним разбиение на предложения точнее.
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" || true

COPY . .

# shyndyq.db (SQLite, см. storage.py) должен жить на volume, а не внутри
# слоя образа - иначе данные студентов терялись бы при каждой пересборке
# образа, что как раз то, от чего мы уходили, добавляя постоянное хранилище.
VOLUME ["/app/data-volume"]
ENV SHYNDYQ_DB_PATH=/app/data-volume/shyndyq.db

EXPOSE 5000 5001

# По умолчанию поднимает веб-интерфейс; для api_analyze.py см. docker-compose.yml
# (там для api-сервиса command переопределён).
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "5000", "--model-dir", "model_multiscale"]
