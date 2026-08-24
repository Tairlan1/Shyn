#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
doc_extract.py
=====================================================================
Извлечение обычного текста из загруженного студентом файла (.docx,
.pdf, .txt) с сохранением разбиения на абзацы (пустая строка между
абзацами) - это важно для стилометрических признаков модели
(длина абзаца, доля диалога и т.д. считаются по абзацам).
"""

from __future__ import annotations

import io


def extract_text(filename: str, raw_bytes: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".docx"):
        return _extract_docx(raw_bytes)
    if name.endswith(".pdf"):
        return _extract_pdf(raw_bytes)
    # .txt и всё прочее - трактуем как обычный текст
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("utf-8", errors="replace")


def _extract_docx(raw_bytes: bytes) -> str:
    import docx  # python-docx

    document = docx.Document(io.BytesIO(raw_bytes))
    paragraphs = [p.text.strip() for p in document.paragraphs]
    paragraphs = [p for p in paragraphs if p]
    return "\n\n".join(paragraphs)


def _extract_pdf(raw_bytes: bytes) -> str:
    import pdfplumber

    pages_text = []
    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(text.strip())
    # pdfplumber обычно сохраняет одиночные переводы строк внутри абзаца
    # и не всегда ставит пустую строку между абзацами - нормализуем так,
    # чтобы соседние строки одного абзаца схлопывались, а разрывы страниц
    # трактовались как границы абзацев.
    normalized = []
    for page_text in pages_text:
        lines = [ln.strip() for ln in page_text.split("\n")]
        lines = [ln for ln in lines if ln]
        normalized.append(" ".join(lines))
    return "\n\n".join(normalized)
