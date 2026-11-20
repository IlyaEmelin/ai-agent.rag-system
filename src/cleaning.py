"""Утилиты очистки и нормализации текстов корпуса «Фильмоскоп»."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

import trafilatura
from bs4 import BeautifulSoup
from ftfy import fix_text

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
MOJIBAKE_MARKERS = ('�', 'Ã', 'Â', 'Ð', 'Ñ', 'â€', 'â€™', 'â€œ', 'â€\\u009d', 'â€“', 'â€”')
SUSPICIOUS_SEQUENCE_PATTERNS = (
    re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"),
    re.compile(r"[\ufffd]"),
    re.compile(r"[!?.,;:]{4,}"),
    re.compile(r"[-_=*#]{6,}"),
    re.compile(r"\s{5,}"),
)
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HYPHENATED_LINE_BREAK_PATTERN = re.compile(r"(?<=[A-Za-zА-Яа-яЁё])[-‐‑‒–—]\n(?=[A-Za-zА-Яа-яЁё])")
BOILERPLATE_TAGS = (
    "script",
    "style",
    "noscript",
    "template",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "iframe",
    "svg",
    "canvas",
)
BOILERPLATE_ATTRIBUTE_TOKENS = (
    "ads",
    "advert",
    "banner",
    "breadcrumb",
    "button",
    "carousel",
    "comment",
    "cookie",
    "feedback",
    "footer",
    "header",
    "login",
    "menu",
    "modal",
    "nav",
    "pagination",
    "popup",
    "promo",
    "rating",
    "related",
    "share",
    "sidebar",
    "social",
    "subscribe",
    "toolbar",
)
MAIN_CONTENT_SELECTORS = (
    "article",
    "main",
    "[role='main']",
    ".article",
    ".content",
    ".doc-content",
    ".entry-content",
    ".kb-article",
    ".main-content",
    ".post-content",
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
    """Собирает диагностические признаки качества текста.

    Аргументы:
        text: Проверяемый текст.

    Возвращает:
        Набор флагов качества текста.
    """

    return TextQualityFlags(
        has_html=has_html_markup(text),
        has_markdown=has_markdown_markup(text),
        mojibake_markers=find_mojibake_markers(text),
        suspicious_sequences=find_suspicious_sequences(text),
    )


def normalize_text_for_duplicate_detection(text: str) -> str:
    """Нормализует текст для поиска дублей без изменения исходного корпуса.

    Аргументы:
        text: Исходный текст документа.

    Возвращает:
        Нижний регистр без разметки и лишних пробелов.
    """

    normalized = html.unescape(text).lower()
    normalized = HTML_TAG_PATTERN.sub(" ", normalized)
    normalized = re.sub(r"[`*_#>|\[\]()]", " ", normalized)
    normalized = re.sub(r"\W+", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def text_similarity(left: str, right: str) -> float:
    """Считает похожесть двух текстов для диагностики почти дублей.

    Аргументы:
        left: Первый нормализованный текст.
        right: Второй нормализованный текст.

    Возвращает:
        Коэффициент похожести ``SequenceMatcher``.
    """

    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(a=left, b=right, autojunk=False).ratio()


def clean_html_markup(text: str) -> str:
    """Извлекает основной читаемый текст из HTML-документа.

    Основной путь использует ``trafilatura``. Если извлечение пустое или слишком
    слабое для короткой/искусственной HTML-страницы, применяется fallback на
    ``BeautifulSoup`` с парсером ``lxml`` и удалением служебных DOM-узлов.

    Аргументы:
        text: HTML-текст документа.

    Возвращает:
        Извлечённый основной текст страницы.
    """

    cleaned, _ = _clean_html_markup_with_method(text)
    return cleaned


def clean_markdown_markup(text: str) -> str:
    """Сохраняет Markdown-структуру и выполняет только безопасную нормализацию.

    На этапе очистки Markdown больше не превращается в plain text: заголовки,
    списки, ссылки, таблицы и inline-code остаются структурными подсказками для
    следующего Markdown-aware чанкинга.

    Аргументы:
        text: Исходный Markdown-текст.

    Возвращает:
        Markdown-текст с едиными переносами строк и без пробелов в конце строк.
    """

    markdown = text.replace("\r\n", "\n").replace("\r", "\n")
    markdown = "\n".join(line.rstrip() for line in markdown.split("\n"))
    return markdown


def fix_mojibake_if_confident(text: str) -> tuple[str, bool]:
    """Исправляет mojibake через ``ftfy`` только при улучшении диагностик.

    Исправление принимается, если ``ftfy`` уменьшает число mojibake-маркеров,
    улучшает общий диагностический score и не уменьшает число кириллических
    символов. Символ ``�`` не восстанавливается эвристически, если исходные
    байты уже потеряны.

    Аргументы:
        text: Проверяемый текст.

    Возвращает:
        Пару ``(текст, исправлено ли)``.
    """

    original_marker_count = _count_mojibake_markers(text)
    if original_marker_count == 0:
        return text, False

    candidate = fix_text(text)
    if candidate == text:
        return text, False
    if _count_mojibake_markers(candidate) >= original_marker_count:
        return text, False
    if _count_cyrillic_letters(candidate) < _count_cyrillic_letters(text):
        return text, False
    if _mojibake_repair_score(candidate) <= _mojibake_repair_score(text):
        return text, False
    return candidate, True


def normalize_whitespace(text: str) -> str:
    """Нормализует Unicode, пробелы, переносы строк и переносы по дефису.

    Аргументы:
        text: Текст после форматной очистки.

    Возвращает:
        Текст с едиными переносами строк, без управляющих символов и лишних
        пробелов. Содержательная Markdown-разметка при этом сохраняется.
    """

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00ad", "")
    normalized = CONTROL_CHAR_PATTERN.sub("", normalized)
    normalized = HYPHENATED_LINE_BREAK_PATTERN.sub("", normalized)
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
        cleaned, html_operation = _clean_html_markup_with_method(cleaned)
        operations.extend(("html_cleaned", html_operation))

    if source_format == "markdown" or has_markdown_markup(cleaned):
        preserved_markdown = clean_markdown_markup(cleaned)
        if preserved_markdown != cleaned:
            operations.append("markdown_whitespace_normalized")
        cleaned = preserved_markdown
        operations.append("markdown_marked_for_chunking")

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


def _clean_html_markup_with_method(text: str) -> tuple[str, str]:
    """Извлекает HTML и возвращает текст вместе с названием выбранного метода."""

    trafilatura_text = _extract_html_with_trafilatura(text)
    fallback_text = ""
    if not _is_meaningful_html_extraction(text, trafilatura_text):
        fallback_text = _extract_html_with_beautifulsoup(text)
        if fallback_text:
            return fallback_text, "html_bs4_fallback"
    if trafilatura_text:
        return trafilatura_text, "html_trafilatura"
    fallback_text = fallback_text or _extract_html_with_beautifulsoup(text)
    if fallback_text:
        return fallback_text, "html_bs4_fallback"
    return html.unescape(HTML_TAG_PATTERN.sub(" ", text)), "html_regex_last_resort"


def _extract_html_with_trafilatura(text: str) -> str:
    """Извлекает основной текст HTML через ``trafilatura``."""

    try:
        extracted = trafilatura.extract(
            text,
            include_comments=False,
            include_formatting=False,
            include_images=False,
            include_links=False,
            include_tables=True,
            no_fallback=False,
            output_format="txt",
        )
    except (LookupError, ValueError, TypeError):
        return ""
    return extracted.strip() if extracted else ""


def _extract_html_with_beautifulsoup(text: str) -> str:
    """Извлекает HTML через ``BeautifulSoup``/``lxml`` после удаления boilerplate."""

    soup = BeautifulSoup(text, "lxml")
    for tag in soup(BOILERPLATE_TAGS):
        tag.decompose()
    for element in list(soup.find_all(True)):
        if _is_boilerplate_element(element):
            element.decompose()

    candidates = [element.get_text("\n", strip=True) for selector in MAIN_CONTENT_SELECTORS for element in soup.select(selector)]
    candidates = [candidate for candidate in candidates if candidate]
    if candidates:
        return max(candidates, key=_content_score).strip()

    body = soup.body or soup
    return body.get_text("\n", strip=True).strip()


def _is_boilerplate_element(element: object) -> bool:
    """Проверяет DOM-элемент на служебные классы, id и ARIA-описания."""

    attrs = getattr(element, "attrs", {}) or {}
    values: list[str] = []
    for attr_name in ("id", "class", "role", "aria-label", "data-testid"):
        value = attrs.get(attr_name)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, Sequence):
            values.extend(str(item) for item in value)
    haystack = " ".join(values).lower()
    return any(token in haystack for token in BOILERPLATE_ATTRIBUTE_TOKENS)


def _is_meaningful_html_extraction(original_html: str, extracted_text: str) -> bool:
    """Оценивает, достаточно ли хорош результат ``trafilatura``."""

    if not extracted_text:
        return False
    word_count = len(re.findall(r"\w+", extracted_text, flags=re.UNICODE))
    if word_count >= 12:
        return True
    visible_text = _extract_html_with_beautifulsoup(original_html)
    return bool(visible_text and len(extracted_text) >= min(80, len(visible_text) * 0.5))


def _content_score(text: str) -> int:
    """Оценивает насыщенность кандидата содержательным текстом."""

    boilerplate_penalty = sum(text.lower().count(token) for token in BOILERPLATE_ATTRIBUTE_TOKENS) * 30
    return len(text) - boilerplate_penalty


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
