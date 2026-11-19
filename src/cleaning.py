"""Утилиты очистки и нормализации текстов корпуса «Фильмоскоп»."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser
from typing import Sequence

from src.data_loading import KnowledgeDocument


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
BLOCK_HTML_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
)
MARKDOWN_LINK_PATTERN = re.compile(r"!?\[([^\]\n]*)\]\(([^\)\n]+)\)")
MARKDOWN_HEADING_PATTERN = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")
MARKDOWN_LIST_PATTERN = re.compile(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)")
MARKDOWN_BLOCKQUOTE_PATTERN = re.compile(r"(?m)^\s{0,3}>\s?")
MARKDOWN_TABLE_SEPARATOR_PATTERN = re.compile(r"(?m)^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
MARKDOWN_DECORATION_PATTERN = re.compile(r"(?<!\w)(?:\*\*|__)([^\n]+?)(?:\*\*|__)(?!\w)|(?<!\w)(?:\*|_)([^\n]+?)(?:\*|_)(?!\w)")
MARKDOWN_CODE_MARKER_PATTERN = re.compile(r"`{1,3}([^`\n]+)`{1,3}")


class _ReadableHtmlParser(HTMLParser):
    """Минимальный извлекатель читаемого текста из HTML на стандартной библиотеке."""

    def __init__(self) -> None:
        """Инициализирует буфер извлечённого текста."""

        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Добавляет разделитель перед блочными HTML-тегами."""

        if tag.lower() in BLOCK_HTML_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Добавляет разделитель после блочных HTML-тегов."""

        if tag.lower() in BLOCK_HTML_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        """Сохраняет текстовые данные HTML-документа."""

        if data:
            self._parts.append(data)

    def get_text(self) -> str:
        """Возвращает извлечённый текст."""

        return "".join(self._parts)


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


@dataclass(frozen=True)
class CleanedDocumentResult:
    """Результат очистки одного документа корпуса.

    Атрибуты:
        original_document: Исходный документ без изменений.
        cleaned_document: Документ с очищенным текстом в поле ``raw``.
        operations: Выполненные операции очистки.
        raw_length_before: Длина исходного текста.
        raw_length_after: Длина очищенного текста.
    """

    original_document: KnowledgeDocument
    cleaned_document: KnowledgeDocument
    operations: tuple[str, ...]
    raw_length_before: int
    raw_length_after: int


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


def clean_html_markup(text: str) -> str:
    """Извлекает читаемый текст из HTML без потери смысловых блоков.

    Аргументы:
        text: Исходный текст, возможно содержащий HTML-теги и entities.

    Возвращает:
        Текст без HTML-тегов, с декодированными HTML entities и сохранёнными
        разделителями между блочными элементами.
    """

    parser = _ReadableHtmlParser()
    parser.feed(text)
    parser.close()
    extracted = parser.get_text()
    if not extracted.strip() and text.strip():
        extracted = HTML_TAG_PATTERN.sub(" ", text)
    return html.unescape(extracted)


def clean_markdown_markup(text: str) -> str:
    """Убирает служебные символы Markdown, сохраняя смысл текста.

    Аргументы:
        text: Исходный текст, возможно содержащий Markdown-разметку.

    Возвращает:
        Текст без основных Markdown-маркеров заголовков, списков, ссылок,
        выделения, inline-code и таблиц.
    """

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"(?m)^```.*$", "", cleaned)
    cleaned = MARKDOWN_LINK_PATTERN.sub(lambda match: match.group(1) or match.group(2), cleaned)
    cleaned = MARKDOWN_TABLE_SEPARATOR_PATTERN.sub("", cleaned)
    cleaned = MARKDOWN_HEADING_PATTERN.sub("", cleaned)
    cleaned = MARKDOWN_LIST_PATTERN.sub("", cleaned)
    cleaned = MARKDOWN_BLOCKQUOTE_PATTERN.sub("", cleaned)
    cleaned = MARKDOWN_CODE_MARKER_PATTERN.sub(lambda match: match.group(1), cleaned)
    cleaned = MARKDOWN_DECORATION_PATTERN.sub(lambda match: match.group(1) or match.group(2) or "", cleaned)
    cleaned = re.sub(r"(?m)^\s*\|(.+)\|\s*$", lambda match: " ".join(part.strip() for part in match.group(1).split("|")), cleaned)
    return cleaned


def fix_mojibake_if_confident(text: str) -> tuple[str, bool]:
    """Исправляет mojibake только при уверенном улучшении текста.

    Эвристика применяет обратное преобразование для частого случая, когда UTF-8
    байты были ошибочно прочитаны как однобайтовая кодировка. Исправление
    принимается только если после него становится меньше mojibake-маркеров и
    больше кириллических символов.

    Аргументы:
        text: Проверяемый текст.

    Возвращает:
        Пару ``(текст, исправлено ли)``.
    """

    original_marker_count = _count_mojibake_markers(text)
    if original_marker_count == 0:
        return text, False

    best_candidate = text
    best_score = _mojibake_repair_score(text)
    for source_encoding in ("latin1", "cp1252"):
        try:
            candidate = text.encode(source_encoding).decode("utf-8")
        except UnicodeError:
            continue
        candidate_score = _mojibake_repair_score(candidate)
        if candidate_score > best_score:
            best_candidate = candidate
            best_score = candidate_score

    if best_candidate == text:
        return text, False
    if _count_mojibake_markers(best_candidate) >= original_marker_count:
        return text, False
    if _count_cyrillic_letters(best_candidate) <= _count_cyrillic_letters(text):
        return text, False
    return best_candidate, True


def normalize_whitespace(text: str) -> str:
    """Нормализует пробелы и переносы строк в очищенном тексте.

    Аргументы:
        text: Текст после исправления кодировки и очистки разметки.

    Возвращает:
        Текст с едиными переносами строк, без лишних пробелов и пустых строк.
    """

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[\t\f\v]+", " ", normalized)
    normalized = re.sub(r"[ \u00a0]{2,}", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def clean_text(text: str, *, source_format: str = "text") -> tuple[str, tuple[str, ...]]:
    """Очищает один текст корпуса и возвращает список выполненных операций.

    Аргументы:
        text: Исходный текст документа.
        source_format: Исходный формат документа из датасета.

    Возвращает:
        Пару ``(очищенный текст, операции)``.
    """

    operations: list[str] = []
    cleaned = text

    repaired, was_repaired = fix_mojibake_if_confident(cleaned)
    if was_repaired:
        cleaned = repaired
        operations.append("mojibake_fixed")

    if source_format == "html" or has_html_markup(cleaned):
        cleaned = clean_html_markup(cleaned)
        operations.append("html_cleaned")

    if source_format == "markdown" or has_markdown_markup(cleaned):
        cleaned = clean_markdown_markup(cleaned)
        operations.append("markdown_cleaned")

    normalized = normalize_whitespace(cleaned)
    if normalized != cleaned:
        operations.append("whitespace_normalized")
    cleaned = normalized

    return cleaned, tuple(dict.fromkeys(operations))


def clean_document(document: KnowledgeDocument) -> CleanedDocumentResult:
    """Очищает один документ базы знаний.

    Аргументы:
        document: Исходный документ корпуса.

    Возвращает:
        Результат очистки с исходным и очищенным документом.
    """

    cleaned_text, operations = clean_text(document.raw, source_format=document.format)
    cleaned_document = KnowledgeDocument(
        doc_id=document.doc_id,
        title=document.title,
        category=document.category,
        format=document.format,
        raw=cleaned_text,
    )
    return CleanedDocumentResult(
        original_document=document,
        cleaned_document=cleaned_document,
        operations=operations,
        raw_length_before=len(document.raw),
        raw_length_after=len(cleaned_text),
    )


def clean_corpus(documents: Sequence[KnowledgeDocument]) -> list[CleanedDocumentResult]:
    """Очищает корпус документов без удаления записей.

    Аргументы:
        documents: Документы базы знаний.

    Возвращает:
        Список результатов очистки в исходном порядке документов.
    """

    return [clean_document(document) for document in documents]


def _count_mojibake_markers(text: str) -> int:
    """Считает суммарное число mojibake-маркеров в тексте."""

    return sum(text.count(marker) for marker in MOJIBAKE_MARKERS)


def _count_cyrillic_letters(text: str) -> int:
    """Считает кириллические буквы в тексте."""

    return sum(1 for char in text if "CYRILLIC" in unicodedata.name(char, ""))


def _mojibake_repair_score(text: str) -> int:
    """Оценивает качество текста для выбора уверенного исправления mojibake."""

    return _count_cyrillic_letters(text) * 3 - _count_mojibake_markers(text) * 20 - text.count("�") * 50


__all__ = [
    "CleanedDocumentResult",
    "DuplicateMatch",
    "TextQualityFlags",
    "clean_corpus",
    "clean_document",
    "clean_html_markup",
    "clean_markdown_markup",
    "clean_text",
    "find_mojibake_markers",
    "find_suspicious_sequences",
    "fix_mojibake_if_confident",
    "get_text_quality_flags",
    "has_html_markup",
    "has_markdown_markup",
    "normalize_whitespace",
    "normalize_text_for_duplicate_detection",
    "text_similarity",
]
