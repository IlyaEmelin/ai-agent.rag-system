"""Утилиты подготовки документов и чанкинга для retrieval-экспериментов."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Sequence

from langchain_core.documents import Document

from src.cleaning import CleanedDocumentResult
from src.data_loading import KnowledgeDocument


REQUIRED_DOCUMENT_METADATA = ("doc_id", "title", "category", "format")
REQUIRED_CHUNK_METADATA = (*REQUIRED_DOCUMENT_METADATA, "chunk_id", "chunk_index", "chunking_strategy")
DEFAULT_RECURSIVE_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")
MARKDOWN_HEADER_PATTERN = re.compile(r"(?m)^\s{0,3}#{1,6}\s+\S")
MARKDOWN_TABLE_ROW_PATTERN = re.compile(r"^\s*\|.+\|\s*$")


@dataclass(frozen=True)
class ChunkingStrategy:
    """Конфигурация одной стратегии чанкинга.

    Атрибуты:
        name: Стабильное имя стратегии для metadata чанка.
        chunk_size: Целевой максимальный размер чанка в символах.
        chunk_overlap: Перекрытие соседних чанков в символах.
        markdown_aware: Нужно ли учитывать Markdown-структуру перед обычным
            рекурсивным делением текста.
    """

    name: str
    chunk_size: int
    chunk_overlap: int
    markdown_aware: bool = False


def build_chunking_strategies_from_settings(chunking_settings: Any) -> tuple[ChunkingStrategy, ChunkingStrategy]:
    """Строит две стратегии чанкинга из блока настроек проекта.

    Аргументы:
        chunking_settings: Объект с полями ``chunk_size``, ``chunk_overlap``,
            ``alt_chunk_size`` и ``alt_chunk_overlap``. Обычно это
            ``settings.chunking`` из ``src.config``.

    Возвращает:
        Пару стратегий: базовую рекурсивную и альтернативную Markdown-aware.
    """

    return (
        ChunkingStrategy(
            name="recursive",
            chunk_size=int(chunking_settings.chunk_size),
            chunk_overlap=int(chunking_settings.chunk_overlap),
            markdown_aware=False,
        ),
        ChunkingStrategy(
            name="markdown_aware",
            chunk_size=int(chunking_settings.alt_chunk_size),
            chunk_overlap=int(chunking_settings.alt_chunk_overlap),
            markdown_aware=True,
        ),
    )


def build_langchain_documents(
    documents: Sequence[KnowledgeDocument | CleanedDocumentResult],
) -> list[Document]:
    """Преобразует документы корпуса в LangChain ``Document``.

    Функция принимает либо уже очищенные ``KnowledgeDocument``, либо результаты
    ``clean_corpus()``. Во втором случае используется поле ``cleaned_document``.
    Это позволяет notebook явно хранить результаты очистки и при этом передавать
    дальше стандартные объекты LangChain.

    Аргументы:
        documents: Последовательность очищенных документов или результатов
            очистки корпуса.

    Возвращает:
        Список ``langchain_core.documents.Document`` с очищенным текстом в
        ``page_content`` и обязательными metadata ``doc_id``, ``title``,
        ``category`` и ``format``. Заголовок и категория остаются в metadata;
        в индексируемый текст каждого чанка они добавляются на этапе
        ``build_document_chunks()``.

    Исключения:
        ValueError: Если документ пустой после очистки или metadata не проходят
            базовую валидацию.
        TypeError: Если элемент последовательности имеет неподдерживаемый тип.
    """

    langchain_documents: list[Document] = []
    for item in documents:
        document = _resolve_knowledge_document(item)
        metadata = build_document_metadata(document)
        page_content = document.raw.strip()
        if not page_content:
            raise ValueError(f"Документ doc_id={document.doc_id} пуст после очистки и не может быть проиндексирован.")
        langchain_documents.append(Document(page_content=page_content, metadata=metadata))
    return langchain_documents


def build_document_metadata(document: KnowledgeDocument) -> dict[str, Any]:
    """Строит metadata для LangChain ``Document`` из документа базы знаний.

    Аргументы:
        document: Документ корпуса «Фильмоскоп».

    Возвращает:
        Словарь metadata с полями ``doc_id``, ``title``, ``category`` и
        ``format``.

    Исключения:
        ValueError: Если обязательные metadata пустые или имеют неверный тип.
    """

    metadata: dict[str, Any] = {
        "doc_id": document.doc_id,
        "title": document.title,
        "category": document.category,
        "format": document.format,
    }
    validate_document_metadata(metadata)
    return metadata


def validate_document_metadata(metadata: dict[str, Any]) -> None:
    """Проверяет metadata документа перед индексацией и расчётом метрик.

    Аргументы:
        metadata: Словарь metadata одного документа.

    Исключения:
        ValueError: Если отсутствует обязательное поле или значение некорректно.
    """

    missing_fields = [field for field in REQUIRED_DOCUMENT_METADATA if field not in metadata]
    if missing_fields:
        raise ValueError(f"В metadata отсутствуют обязательные поля: {', '.join(missing_fields)}.")

    doc_id = metadata["doc_id"]
    if not isinstance(doc_id, int):
        raise ValueError(f"metadata.doc_id должен быть целым числом, получено: {doc_id!r}.")

    for field in ("title", "category", "format"):
        value = metadata[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"metadata.{field} должен быть непустой строкой, получено: {value!r}.")


def summarize_langchain_documents(documents: Sequence[Document]) -> dict[str, Any]:
    """Собирает краткую сводку по подготовленным LangChain ``Document``.

    Аргументы:
        documents: Последовательность подготовленных LangChain-документов.

    Возвращает:
        Словарь с количеством документов, распределением форматов/категорий,
        числом уникальных ``doc_id`` и диапазоном длин ``page_content``.
    """

    lengths = [len(document.page_content) for document in documents]
    doc_ids = [document.metadata.get("doc_id") for document in documents]
    formats = Counter(str(document.metadata.get("format", "")) for document in documents)
    categories = Counter(str(document.metadata.get("category", "")) for document in documents)

    return {
        "documents": len(documents),
        "unique_doc_ids": len(set(doc_ids)),
        "min_page_content_length": min(lengths) if lengths else 0,
        "max_page_content_length": max(lengths) if lengths else 0,
        "formats": dict(sorted(formats.items())),
        "categories": dict(sorted(categories.items())),
    }


def build_document_chunks(documents: Sequence[Document], strategy: ChunkingStrategy) -> list[Document]:
    """Строит чанки LangChain ``Document`` по выбранной стратегии.

    Аргументы:
        documents: Подготовленные документы после ``build_langchain_documents``.
        strategy: Конфигурация стратегии чанкинга.

    Возвращает:
        Список чанков. Каждый чанк наследует metadata исходного документа и
        получает поля ``chunk_id``, ``chunk_index`` и ``chunking_strategy``.

    Исключения:
        ValueError: Если стратегия некорректна, исходный документ пустой или
            metadata чанка не проходят валидацию.
    """

    _validate_chunking_strategy(strategy)
    chunks: list[Document] = []
    for document in documents:
        validate_document_metadata(document.metadata)
        text = document.page_content.strip()
        if not text:
            raise ValueError(f"Документ doc_id={document.metadata.get('doc_id')} пуст и не может быть разбит на чанки.")

        text_chunks = split_document_text(text, document.metadata, strategy)
        for chunk_index, chunk_text in enumerate(text_chunks):
            metadata = build_chunk_metadata(document.metadata, chunk_index, strategy.name)
            validate_chunk_metadata(metadata)
            indexed_chunk_text = build_indexed_chunk_text(chunk_text, metadata)
            chunks.append(Document(page_content=indexed_chunk_text, metadata=metadata))
    return chunks


def build_indexed_chunk_text(chunk_text: str, metadata: dict[str, Any]) -> str:
    """Добавляет поисковый контекст metadata в текст чанка.

    Заголовок и категория важны для ранжирования, но сами по себе Chroma и BM25
    ищут по ``page_content``. Поэтому эти поля дублируются в индексируемый текст
    каждого чанка, при этом исходные значения сохраняются в metadata для
    фильтрации, трассировки и вывода источников.

    Аргументы:
        chunk_text: Текст чанка после выбранной стратегии разбиения.
        metadata: Metadata чанка с полями ``title`` и ``category``.

    Возвращает:
        Текст чанка с префиксом ``Название``/``Категория``/``Текст``.

    Исключения:
        ValueError: Если чанк пустой или metadata не проходят валидацию.
    """

    validate_chunk_metadata(metadata)
    text = chunk_text.strip()
    if not text:
        raise ValueError(f"Чанк doc_id={metadata.get('doc_id')} пуст и не может быть проиндексирован.")

    title = str(metadata["title"]).strip()
    category = str(metadata["category"]).strip()
    return f"Название: {title}\nКатегория: {category}\nТекст:\n{text}"


def split_document_text(text: str, metadata: dict[str, Any], strategy: ChunkingStrategy) -> list[str]:
    """Разбивает текст одного документа согласно стратегии.

    Markdown-aware стратегия сначала делит Markdown-документы по структурным
    границам: заголовкам, пустым строкам и таблицам. Затем каждый полученный
    блок при необходимости дополнительно режется рекурсивной стратегией, чтобы
    соблюдать лимит ``chunk_size``.

    Аргументы:
        text: Текст одного документа.
        metadata: Metadata исходного документа.
        strategy: Конфигурация стратегии чанкинга.

    Возвращает:
        Непустой список текстовых чанков.
    """

    normalized_text = text.strip()
    if not normalized_text:
        return []

    if strategy.markdown_aware and metadata.get("format") == "markdown":
        markdown_sections = split_markdown_sections(normalized_text)
        return _split_sections_with_recursive_fallback(markdown_sections, strategy)

    return recursive_split_text(normalized_text, strategy.chunk_size, strategy.chunk_overlap)


def build_chunk_metadata(document_metadata: dict[str, Any], chunk_index: int, chunking_strategy: str) -> dict[str, Any]:
    """Создаёт metadata чанка на основе metadata исходного документа.

    Аргументы:
        document_metadata: Metadata исходного LangChain-документа.
        chunk_index: Порядковый номер чанка внутри исходного документа.
        chunking_strategy: Имя стратегии чанкинга.

    Возвращает:
        Metadata чанка с обязательными полями для retrieval и диагностики.
    """

    doc_id = document_metadata["doc_id"]
    metadata = dict(document_metadata)
    metadata.update(
        {
            "chunk_id": f"{doc_id}:{chunking_strategy}:{chunk_index}",
            "chunk_index": chunk_index,
            "chunking_strategy": chunking_strategy,
        }
    )
    return metadata


def validate_chunk_metadata(metadata: dict[str, Any]) -> None:
    """Проверяет metadata чанка перед индексацией.

    Аргументы:
        metadata: Metadata одного чанка.

    Исключения:
        ValueError: Если отсутствует обязательное поле или значение некорректно.
    """

    validate_document_metadata(metadata)
    missing_fields = [field for field in REQUIRED_CHUNK_METADATA if field not in metadata]
    if missing_fields:
        raise ValueError(f"В metadata чанка отсутствуют обязательные поля: {', '.join(missing_fields)}.")

    chunk_id = metadata["chunk_id"]
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError(f"metadata.chunk_id должен быть непустой строкой, получено: {chunk_id!r}.")
    chunk_index = metadata["chunk_index"]
    if not isinstance(chunk_index, int) or chunk_index < 0:
        raise ValueError(f"metadata.chunk_index должен быть неотрицательным целым числом, получено: {chunk_index!r}.")
    chunking_strategy = metadata["chunking_strategy"]
    if not isinstance(chunking_strategy, str) or not chunking_strategy.strip():
        raise ValueError(
            f"metadata.chunking_strategy должен быть непустой строкой, получено: {chunking_strategy!r}."
        )


def summarize_document_chunks(chunks: Sequence[Document]) -> dict[str, Any]:
    """Собирает сводку по построенным чанкам.

    Аргументы:
        chunks: Последовательность чанков LangChain ``Document``.

    Возвращает:
        Словарь с количеством чанков, распределениями metadata и статистикой
        длины ``page_content``.
    """

    lengths = [len(chunk.page_content) for chunk in chunks]
    doc_ids = [chunk.metadata.get("doc_id") for chunk in chunks]
    strategies = Counter(str(chunk.metadata.get("chunking_strategy", "")) for chunk in chunks)
    formats = Counter(str(chunk.metadata.get("format", "")) for chunk in chunks)
    categories = Counter(str(chunk.metadata.get("category", "")) for chunk in chunks)

    return {
        "chunks": len(chunks),
        "source_documents": len(set(doc_ids)),
        "min_chunk_length": min(lengths) if lengths else 0,
        "max_chunk_length": max(lengths) if lengths else 0,
        "avg_chunk_length": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "strategies": dict(sorted(strategies.items())),
        "formats": dict(sorted(formats.items())),
        "categories": dict(sorted(categories.items())),
    }


def _resolve_knowledge_document(item: KnowledgeDocument | CleanedDocumentResult) -> KnowledgeDocument:
    """Возвращает очищенный ``KnowledgeDocument`` из поддерживаемого входа."""

    if isinstance(item, CleanedDocumentResult):
        return item.cleaned_document
    if isinstance(item, KnowledgeDocument):
        return item
    raise TypeError(f"Ожидался KnowledgeDocument или CleanedDocumentResult, получено: {type(item)!r}.")


def recursive_split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Рекурсивно разбивает текст по убывающим по качеству разделителям.

    Аргументы:
        text: Исходный текст.
        chunk_size: Целевой максимальный размер чанка в символах.
        chunk_overlap: Перекрытие соседних чанков в символах.

    Возвращает:
        Непустой список чанков без пустых строк.
    """

    _validate_chunking_numbers(chunk_size, chunk_overlap)
    stripped_text = text.strip()
    if not stripped_text:
        return []
    if len(stripped_text) <= chunk_size:
        return [stripped_text]

    atomic_parts = _recursive_split_to_atomic_parts(stripped_text, chunk_size, DEFAULT_RECURSIVE_SEPARATORS)
    return _merge_atomic_parts(atomic_parts, chunk_size, chunk_overlap)


def split_markdown_sections(text: str) -> list[str]:
    """Делит Markdown-текст на структурные секции.

    Граница секции создаётся перед Markdown-заголовками, после пустых строк для
    уже достаточно крупных блоков и вокруг таблиц. Это сохраняет заголовки,
    списки и таблицы как подсказки для дальнейшего retrieval.

    Аргументы:
        text: Markdown-текст после очистки.

    Возвращает:
        Список непустых секций.
    """

    sections: list[str] = []
    current_lines: list[str] = []
    inside_table = False

    for line in text.splitlines():
        is_header = bool(MARKDOWN_HEADER_PATTERN.match(line))
        is_table_row = bool(MARKDOWN_TABLE_ROW_PATTERN.match(line))

        if is_header and current_lines:
            _append_section(sections, current_lines)
            current_lines = []

        if inside_table and not is_table_row and current_lines:
            _append_section(sections, current_lines)
            current_lines = []
        inside_table = is_table_row

        current_lines.append(line)
        if not line.strip() and _section_length(current_lines) >= 250:
            _append_section(sections, current_lines)
            current_lines = []
            inside_table = False

    _append_section(sections, current_lines)
    return sections or [text.strip()]


def _split_sections_with_recursive_fallback(sections: Sequence[str], strategy: ChunkingStrategy) -> list[str]:
    """Делит крупные секции и объединяет мелкие с заданным overlap."""

    atomic_parts: list[str] = []
    for section in sections:
        stripped_section = section.strip()
        if not stripped_section:
            continue
        if len(stripped_section) <= strategy.chunk_size:
            atomic_parts.append(stripped_section)
        else:
            atomic_parts.extend(recursive_split_text(stripped_section, strategy.chunk_size, strategy.chunk_overlap))
    return _merge_atomic_parts(atomic_parts, strategy.chunk_size, strategy.chunk_overlap)


def _recursive_split_to_atomic_parts(text: str, chunk_size: int, separators: Sequence[str]) -> list[str]:
    """Рекурсивно получает части, не превышающие ``chunk_size``."""

    stripped_text = text.strip()
    if len(stripped_text) <= chunk_size:
        return [stripped_text]
    if not separators:
        return [stripped_text[index : index + chunk_size].strip() for index in range(0, len(stripped_text), chunk_size)]

    separator = separators[0]
    if separator == "":
        return [stripped_text[index : index + chunk_size].strip() for index in range(0, len(stripped_text), chunk_size)]

    raw_parts = stripped_text.split(separator)
    if len(raw_parts) == 1:
        return _recursive_split_to_atomic_parts(stripped_text, chunk_size, separators[1:])

    atomic_parts: list[str] = []
    for index, raw_part in enumerate(raw_parts):
        part = raw_part.strip()
        if not part:
            continue
        if index < len(raw_parts) - 1:
            part = f"{part}{separator.rstrip()}".strip()
        if len(part) <= chunk_size:
            atomic_parts.append(part)
        else:
            atomic_parts.extend(_recursive_split_to_atomic_parts(part, chunk_size, separators[1:]))
    return atomic_parts


def _merge_atomic_parts(parts: Sequence[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Объединяет короткие части в чанки с перекрытием."""

    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    for part in (item.strip() for item in parts if item.strip()):
        separator_length = 1 if current_parts else 0
        projected_length = current_length + separator_length + len(part)
        if current_parts and projected_length > chunk_size:
            chunk = "\n".join(current_parts).strip()
            chunks.append(chunk)
            current_parts, current_length = _build_overlap_prefix(chunk, chunk_overlap)
            if current_parts and current_length + 1 + len(part) > chunk_size:
                current_parts, current_length = [], 0
        current_parts.append(part)
        current_length += (1 if current_length else 0) + len(part)

    if current_parts:
        chunks.append("\n".join(current_parts).strip())
    return [chunk for chunk in chunks if chunk]


def _build_overlap_prefix(previous_chunk: str, chunk_overlap: int) -> tuple[list[str], int]:
    """Формирует начальный фрагмент следующего чанка из конца предыдущего."""

    if chunk_overlap <= 0:
        return [], 0
    overlap_text = previous_chunk[-chunk_overlap:].strip()
    if not overlap_text:
        return [], 0
    return [overlap_text], len(overlap_text)


def _append_section(sections: list[str], lines: list[str]) -> None:
    """Добавляет непустую Markdown-секцию в список."""

    section = "\n".join(lines).strip()
    if section:
        sections.append(section)


def _section_length(lines: Sequence[str]) -> int:
    """Возвращает приблизительную длину секции с переносами строк."""

    return sum(len(line) for line in lines) + max(len(lines) - 1, 0)


def _validate_chunking_strategy(strategy: ChunkingStrategy) -> None:
    """Проверяет конфигурацию стратегии чанкинга."""

    if not isinstance(strategy.name, str) or not strategy.name.strip():
        raise ValueError(f"Имя стратегии чанкинга должно быть непустой строкой, получено: {strategy.name!r}.")
    _validate_chunking_numbers(strategy.chunk_size, strategy.chunk_overlap)


def _validate_chunking_numbers(chunk_size: int, chunk_overlap: int) -> None:
    """Проверяет числовые параметры чанкинга."""

    if chunk_size <= 0:
        raise ValueError(f"chunk_size должен быть положительным числом, получено: {chunk_size}.")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap должен быть неотрицательным числом, получено: {chunk_overlap}.")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap должен быть меньше chunk_size, получено: {chunk_overlap} >= {chunk_size}."
        )


__all__ = [
    "ChunkingStrategy",
    "REQUIRED_CHUNK_METADATA",
    "REQUIRED_DOCUMENT_METADATA",
    "build_chunk_metadata",
    "build_chunking_strategies_from_settings",
    "build_document_chunks",
    "build_document_metadata",
    "build_langchain_documents",
    "recursive_split_text",
    "split_document_text",
    "split_markdown_sections",
    "summarize_document_chunks",
    "summarize_langchain_documents",
    "validate_chunk_metadata",
    "validate_document_metadata",
]