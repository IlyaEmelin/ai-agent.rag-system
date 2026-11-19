"""Утилиты очистки и нормализации текстов корпуса «Фильмоскоп».

На этапе первичного аудита модуль содержит только безопасные эвристики: они не
меняют корпус, а помогают найти документы, которые требуют решения на следующем
этапе очистки.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher


HTML_TAG_PATTERN = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s+[^<>]*)?>")
HTML_ENTITY_PATTERN = re.compile(r"&(?:[a-zA-Z][a-zA-Z0-9]+|#[0-9]+|#x[0-9a-fA-F]+);")
MARKDOWN_PATTERNS = (
    re.compile(r"(?m)^\s{0,3}#{1,6}\s+\S"),
    re.compile(r"(?m)^\s*[-*+]\s+\S"),
    re.compile(r"(?m)^\s*\d+[.)]\s+\S"),
    re.compile(r"\[[^\]\n]+\]\([^\)\n]+\)"),
    re.compile(r"(?:^|\s)(?:\*\*|__)[^\n]+?(?:\*\*|__)(?:\s|$)"),
    re.compile(r"`{1,3}[^`\n]+`{1,3}"),
    re.compile(r"(?m)^\s*\|.+\|\s*$"),
)
MOJIBAKE_MARKERS = ("�", "Ã", "Â", "Ð", "Ñ", "â€", "â€™", "â€œ", "â€\u009d", "â€“", "â€”")
SUSPICIOUS_SEQUENCE_PATTERNS = (
    re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"),
    re.compile(r"[\ufffd]"),
    re.compile(r"[!?.,;:]{4,}"),
    re.compile(r"[-_=*#]{6,}"),
    re.compile(r"\s{5,}"),
)


@dataclass(frozen=True)
class TextQualityFlags:
    """Набор признаков качества исходного текста документа.

    Атрибуты:
        has_html: Есть признаки HTML-тегов или HTML entities.
        has_markdown: Есть признаки Markdown-разметки.
        mojibake_markers: Найденные маркеры ошибок кодировки.
        suspicious_sequences: Найденные мусорные или подозрительные фрагменты.
    """

    has_html: bool
    has_markdown: bool
    mojibake_markers: tuple[str, ...]
    suspicious_sequences: tuple[str, ...]


@dataclass(frozen=True)
class DuplicateMatch:
    """Пара документов с высокой похожестью нормализованного текста.

    Атрибуты:
        left_doc_id: Идентификатор первого документа.
        right_doc_id: Идентификатор второго документа.
        similarity: Коэффициент похожести от ``0`` до ``1``.
        match_type: Тип совпадения: ``exact`` или ``near``.
    """

    left_doc_id: int
    right_doc_id: int
    similarity: float
    match_type: str


def has_html_markup(text: str) -> bool:
    """Проверяет наличие HTML-разметки в тексте.

    Аргументы:
        text: Проверяемый исходный текст.

    Возвращает:
        ``True``, если найдены HTML-теги или HTML entities.
    """

    return bool(HTML_TAG_PATTERN.search(text) or HTML_ENTITY_PATTERN.search(text))


def has_markdown_markup(text: str) -> bool:
    """Проверяет наличие Markdown-разметки в тексте.

    Аргументы:
        text: Проверяемый исходный текст.

    Возвращает:
        ``True``, если текст похож на Markdown-документ.
    """

    return any(pattern.search(text) for pattern in MARKDOWN_PATTERNS)


def find_mojibake_markers(text: str) -> tuple[str, ...]:
    """Находит признаки mojibake и ошибок кодировки.

    Аргументы:
        text: Проверяемый исходный текст.

    Возвращает:
        Уникальные маркеры mojibake в порядке объявления эвристик.
    """

    return tuple(marker for marker in MOJIBAKE_MARKERS if marker in text)


def find_suspicious_sequences(text: str, *, max_examples: int = 5) -> tuple[str, ...]:
    """Находит мусорные символы и подозрительные последовательности.

    Аргументы:
        text: Проверяемый исходный текст.
        max_examples: Максимальное число примеров в результате.

    Возвращает:
        Кортеж найденных фрагментов без дублей.
    """

    examples: list[str] = []
    seen: set[str] = set()
    for pattern in SUSPICIOUS_SEQUENCE_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0)
            if value not in seen:
                examples.append(value)
                seen.add(value)
            if len(examples) >= max_examples:
                return tuple(examples)
    return tuple(examples)


def get_text_quality_flags(text: str) -> TextQualityFlags:
    """Собирает признаки качества текста документа.

    Аргументы:
        text: Проверяемый исходный текст.

    Возвращает:
        Набор флагов и примеров подозрительных фрагментов.
    """

    return TextQualityFlags(
        has_html=has_html_markup(text),
        has_markdown=has_markdown_markup(text),
        mojibake_markers=find_mojibake_markers(text),
        suspicious_sequences=find_suspicious_sequences(text),
    )


def normalize_text_for_duplicate_detection(text: str) -> str:
    """Нормализует текст для поиска полных и почти полных дублей.

    Нормализация намеренно не используется как очистка корпуса: она нужна только
    для сравнения документов в аудите.

    Аргументы:
        text: Исходный текст документа.

    Возвращает:
        Упрощённое строковое представление для duplicate detection.
    """

    unescaped = html.unescape(text)
    normalized = unicodedata.normalize("NFKC", unescaped).casefold()
    normalized = HTML_TAG_PATTERN.sub(" ", normalized)
    normalized = re.sub(r"[`*_#>\[\](){}|]", " ", normalized)
    normalized = re.sub(r"\W+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def text_similarity(left: str, right: str) -> float:
    """Считает похожесть двух уже нормализованных текстов.

    Аргументы:
        left: Первый нормализованный текст.
        right: Второй нормализованный текст.

    Возвращает:
        Коэффициент похожести ``SequenceMatcher`` от ``0`` до ``1``.
    """

    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(a=left, b=right, autojunk=False).ratio()


__all__ = [
    "DuplicateMatch",
    "TextQualityFlags",
    "find_mojibake_markers",
    "find_suspicious_sequences",
    "get_text_quality_flags",
    "has_html_markup",
    "has_markdown_markup",
    "normalize_text_for_duplicate_detection",
    "text_similarity",
]
