"""Вспомогательные функции векторного хранилища для локального прототипа Chroma."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, Sequence

from langchain_core.documents import Document

from src.config import ChromaSettings


@dataclass(frozen=True)
class ChromaCollectionInfo:
    """Сводная информация о временной Chroma collection.

    Атрибуты:
        collection_name: Имя collection в Chroma.
        persist_dir: Директория хранения Chroma-данных.
        document_count: Количество добавленных чанков.
    """

    collection_name: str
    persist_dir: Path
    document_count: int


def build_main_collection_name(prefix: str, strategy_name: str) -> str:
    """Строит имя основной Chroma collection для выбранной стратегии чанкинга.

    Аргументы:
        prefix: Префикс из настроек ``CHROMA_COLLECTION_PREFIX``.
        strategy_name: Имя выбранной стратегии чанкинга.

    Возвращает:
        Стабильное имя основной collection.
    """

    return build_collection_name(prefix, strategy_name, suffix="main")


def build_collection_name(prefix: str, strategy_name: str, *, suffix: str = "tmp_compare") -> str:
    """Строит стабильное имя collection для эксперимента сравнения чанкинга.

    Аргументы:
        prefix: Префикс из настроек ``CHROMA_COLLECTION_PREFIX``.
        strategy_name: Имя стратегии чанкинга.
        suffix: Суффикс эксперимента, отделяющий временные коллекции от основных.

    Возвращает:
        Имя collection, пригодное для Chroma.
    """

    parts = [prefix.strip(), suffix.strip(), strategy_name.strip()]
    normalized = "_".join(part for part in parts if part)
    if not normalized:
        raise ValueError("Имя Chroma collection не может быть пустым.")
    return normalized.replace("-", "_")


def recreate_chroma_collection(
    settings: ChromaSettings,
    collection_name: str,
    documents: Sequence[Document],
    embeddings: Sequence[Sequence[float]],
) -> Any:
    """Пересоздаёт Chroma collection и добавляет чанки с готовыми embeddings.

    Функция используется для воспроизводимых временных экспериментов: если
    collection с таким именем уже существует, она удаляется перед повторным
    добавлением документов.

    Аргументы:
        settings: Блок ``settings.chroma`` из конфигурации проекта.
        collection_name: Имя Chroma collection.
        documents: Чанки LangChain ``Document``.
        embeddings: Готовые embedding-векторы в том же порядке, что и чанки.

    Возвращает:
        Объект ``chromadb.api.models.Collection.Collection``.

    Исключения:
        ImportError: Если пакет ``chromadb`` не установлен.
        ValueError: Если входные данные некорректны.
    """

    _validate_collection_inputs(collection_name, documents, embeddings)
    client = build_persistent_chroma_client(settings.persist_dir)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    add_documents_with_embeddings(collection, documents, embeddings)
    return collection


def recreate_main_chroma_collection(
    settings: ChromaSettings,
    strategy_name: str,
    documents: Sequence[Document],
    embeddings: Sequence[Sequence[float]],
) -> Any:
    """Пересоздаёт основную Chroma collection для baseline retrieval.

    Функция использует ``CHROMA_PERSIST_DIR`` и ``CHROMA_COLLECTION_PREFIX`` из
    настроек проекта. Повторный запуск сначала удаляет collection с тем же
    именем, поэтому Chroma не накапливает неконтролируемые дубли чанков.

    Аргументы:
        settings: Блок ``settings.chroma`` из конфигурации проекта.
        strategy_name: Имя выбранной стратегии чанкинга.
        documents: Чанки LangChain ``Document``.
        embeddings: Готовые embedding-векторы чанков.

    Возвращает:
        Пересозданная Chroma collection.
    """

    collection_name = build_main_collection_name(settings.collection_prefix, strategy_name)
    return recreate_chroma_collection(settings, collection_name, documents, embeddings)


def build_persistent_chroma_client(persist_dir: str | Path) -> Any:
    """Создаёт persistent Chroma client для локального прототипа.

    Аргументы:
        persist_dir: Директория хранения Chroma-данных.

    Возвращает:
        Экземпляр ``chromadb.PersistentClient``.
    """

    try:
        import chromadb
    except ImportError as exc:
        raise ImportError("Пакет chromadb не установлен. Установите зависимости командой: pip install -r requirements.txt") from exc

    path = Path(persist_dir)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def add_documents_with_embeddings(
    collection: Any,
    documents: Sequence[Document],
    embeddings: Sequence[Sequence[float]],
    *,
    batch_size: int = 256,
) -> None:
    """Добавляет чанки и готовые embeddings в Chroma collection.

    Аргументы:
        collection: Chroma collection.
        documents: Чанки LangChain ``Document``.
        embeddings: Embedding-векторы чанков.
        batch_size: Размер batch для добавления в Chroma.
    """

    _validate_collection_inputs(getattr(collection, "name", "collection"), documents, embeddings)
    if batch_size <= 0:
        raise ValueError(f"batch_size должен быть положительным числом, получено: {batch_size}.")

    ids = [str(document.metadata["chunk_id"]) for document in documents]
    texts = [document.page_content for document in documents]
    metadatas = [dict(document.metadata) for document in documents]
    vectors = [list(vector) for vector in embeddings]
    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=vectors[start:end],
        )


def describe_chroma_collection(collection: Any, persist_dir: str | Path) -> ChromaCollectionInfo:
    """Возвращает краткую сводку по Chroma collection.

    Аргументы:
        collection: Chroma collection.
        persist_dir: Директория хранения Chroma-данных.

    Возвращает:
        ``ChromaCollectionInfo`` с именем collection и количеством записей.
    """

    return ChromaCollectionInfo(
        collection_name=str(collection.name),
        persist_dir=Path(persist_dir),
        document_count=int(collection.count()),
    )


def peek_chroma_collection_records(collection: Any, *, limit: int = 3, text_preview_chars: int = 160) -> list[dict[str, Any]]:
    """Возвращает несколько записей collection для проверки документов и metadata.

    Аргументы:
        collection: Chroma collection.
        limit: Максимальное число записей для просмотра.
        text_preview_chars: Максимальная длина текстового превью.

    Возвращает:
        Список словарей с ``id``, ``metadata`` и укороченным текстом чанка.
    """

    normalized_limit = int(limit)
    if normalized_limit <= 0:
        raise ValueError(f"limit должен быть положительным числом, получено: {limit}.")
    normalized_preview = int(text_preview_chars)
    if normalized_preview <= 0:
        raise ValueError(f"text_preview_chars должен быть положительным числом, получено: {text_preview_chars}.")

    response = collection.get(limit=normalized_limit, include=["documents", "metadatas"])
    ids = response.get("ids") or []
    documents = response.get("documents") or []
    metadatas = response.get("metadatas") or []
    records: list[dict[str, Any]] = []
    for record_id, text, metadata in zip(ids, documents, metadatas):
        text_value = str(text or "")
        records.append(
            {
                "id": str(record_id),
                "metadata": dict(metadata or {}),
                "text_preview": _trim_text(text_value, normalized_preview),
            }
        )
    return records


def remove_chroma_persist_dir(persist_dir: str | Path) -> None:
    """Удаляет директорию временного Chroma-хранилища, если она существует.

    На Windows Chroma может кратковременно удерживать файлы segment-хранилища
    открытыми даже после завершения операций с collection. Для временных
    экспериментальных директорий такая ошибка не должна ломать notebook:
    неудалённые файлы будут перезаписаны или очищены следующим запуском.

    Аргументы:
        persist_dir: Директория временных Chroma-данных.
    """

    path = Path(persist_dir)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _validate_collection_inputs(
    collection_name: str,
    documents: Sequence[Document],
    embeddings: Sequence[Sequence[float]],
) -> None:
    """Проверяет входные данные перед добавлением в Chroma."""

    if not isinstance(collection_name, str) or not collection_name.strip():
        raise ValueError(f"Имя Chroma collection должно быть непустой строкой, получено: {collection_name!r}.")
    if not documents:
        raise ValueError("Список документов для Chroma не должен быть пустым.")
    if len(documents) != len(embeddings):
        raise ValueError(f"Количество документов и embeddings должно совпадать: {len(documents)} != {len(embeddings)}.")

    dimensions = {len(vector) for vector in embeddings}
    if len(dimensions) != 1:
        raise ValueError(f"Все embedding-векторы должны иметь одинаковую размерность, получено: {sorted(dimensions)}.")

    chunk_ids: set[str] = set()
    for index, document in enumerate(documents):
        chunk_id = document.metadata.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"У документа index={index} отсутствует непустой metadata.chunk_id.")
        if chunk_id in chunk_ids:
            raise ValueError(f"Найден повторяющийся metadata.chunk_id: {chunk_id}.")
        chunk_ids.add(chunk_id)
        if not isinstance(document.page_content, str) or not document.page_content.strip():
            raise ValueError(f"Документ index={index}, chunk_id={chunk_id} имеет пустой page_content.")


def _trim_text(text: str, max_chars: int) -> str:
    """Обрезает текст до заданной длины для диагностических таблиц."""

    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 1]}…"


__all__ = [
    "ChromaCollectionInfo",
    "add_documents_with_embeddings",
    "build_collection_name",
    "build_main_collection_name",
    "build_persistent_chroma_client",
    "describe_chroma_collection",
    "peek_chroma_collection_records",
    "recreate_chroma_collection",
    "recreate_main_chroma_collection",
    "remove_chroma_persist_dir",
]