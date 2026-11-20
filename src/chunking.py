"""Утилиты подготовки документов и чанкинга для retrieval-экспериментов."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from langchain_core.documents import Document

from src.cleaning import CleanedDocumentResult
from src.data_loading import KnowledgeDocument


REQUIRED_DOCUMENT_METADATA = ("doc_id", "title", "category", "format")


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
        ``category`` и ``format``.

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


def _resolve_knowledge_document(item: KnowledgeDocument | CleanedDocumentResult) -> KnowledgeDocument:
    """Возвращает очищенный ``KnowledgeDocument`` из поддерживаемого входа."""

    if isinstance(item, CleanedDocumentResult):
        return item.cleaned_document
    if isinstance(item, KnowledgeDocument):
        return item
    raise TypeError(f"Ожидался KnowledgeDocument или CleanedDocumentResult, получено: {type(item)!r}.")


__all__ = [
    "REQUIRED_DOCUMENT_METADATA",
    "build_document_metadata",
    "build_langchain_documents",
    "summarize_langchain_documents",
    "validate_document_metadata",
]