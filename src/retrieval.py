"""Утилиты retrieval-пайплайна для dense, sparse и hybrid поиска."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from src.config import RetrievalSettings
from src.data_loading import EvaluationQuery


@dataclass(frozen=True)
class RetrievedChunk:
    """Один найденный retrieval-результат.

    Атрибуты:
        chunk_id: Идентификатор чанка из metadata.
        doc_id: Идентификатор исходного документа.
        text: Текст найденного чанка.
        metadata: Metadata найденного чанка.
        distance: Расстояние Chroma до запроса, если оно возвращено.
        score: Удобный similarity score ``1 - distance`` для cosine space.
        rank: Позиция результата в выдаче, начиная с 1.
    """

    chunk_id: str
    doc_id: int
    text: str
    metadata: dict[str, Any]
    distance: float | None
    score: float | None
    rank: int


@dataclass(frozen=True)
class RetrievalResult:
    """Результат dense retrieval для одного тестового запроса.

    Атрибуты:
        query: Текст запроса.
        kind: Тип запроса из датасета.
        relevant_docs: Список эталонных ``doc_id``.
        retrieved_chunks: Найденные чанки.
    """

    query: str
    kind: str
    relevant_docs: list[int]
    retrieved_chunks: list[RetrievedChunk]

    @property
    def retrieved_doc_ids(self) -> list[int]:
        """Возвращает ``doc_id`` найденных чанков с сохранением рангов и дублей."""

        return [chunk.doc_id for chunk in self.retrieved_chunks]

    @property
    def unique_retrieved_doc_ids(self) -> list[int]:
        """Возвращает уникальные ``doc_id`` в порядке первого появления."""

        seen: set[int] = set()
        unique_ids: list[int] = []
        for doc_id in self.retrieved_doc_ids:
            if doc_id not in seen:
                unique_ids.append(doc_id)
                seen.add(doc_id)
        return unique_ids


def dense_retrieve(
    collection: Any,
    embedding_model: Any,
    query: str,
    settings: RetrievalSettings,
    *,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Выполняет dense retrieval одного запроса через Chroma.

    Аргументы:
        collection: Chroma collection с чанками и embeddings.
        embedding_model: Wrapper из ``src.embeddings`` с методом ``embed_query``.
        query: Пользовательский вопрос.
        settings: Блок ``settings.retrieval``.
        top_k: Необязательное переопределение числа возвращаемых чанков.

    Возвращает:
        Список найденных чанков с metadata, distance и rank.
    """

    requested_top_k = int(top_k if top_k is not None else settings.top_k)
    if requested_top_k <= 0:
        raise ValueError(f"top_k должен быть положительным числом, получено: {requested_top_k}.")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"Текст запроса должен быть непустой строкой, получено: {query!r}.")

    query_embedding = embedding_model.embed_query(query)
    response = collection.query(
        query_embeddings=[query_embedding],
        n_results=requested_top_k,
        include=["documents", "metadatas", "distances"],
    )
    return _parse_chroma_query_response(response, score_threshold=settings.score_threshold)


def evaluate_dense_retrieval(
    collection: Any,
    embedding_model: Any,
    queries: Sequence[EvaluationQuery],
    settings: RetrievalSettings,
    *,
    top_k: int | None = None,
    include_gap_queries: bool = False,
) -> list[RetrievalResult]:
    """Прогоняет dense retrieval по набору тестовых запросов.

    Аргументы:
        collection: Chroma collection.
        embedding_model: Wrapper embedding-модели.
        queries: Тестовые запросы датасета.
        settings: Настройки retrieval.
        top_k: Необязательное число возвращаемых чанков.
        include_gap_queries: Нужно ли включать ``gap``-запросы в прогон.

    Возвращает:
        Список ``RetrievalResult``.
    """

    results: list[RetrievalResult] = []
    for evaluation_query in queries:
        if evaluation_query.kind == "gap" and not include_gap_queries:
            continue
        chunks = dense_retrieve(collection, embedding_model, evaluation_query.query, settings, top_k=top_k)
        results.append(
            RetrievalResult(
                query=evaluation_query.query,
                kind=evaluation_query.kind,
                relevant_docs=list(evaluation_query.relevant_docs),
                retrieved_chunks=chunks,
            )
        )
    return results


def _parse_chroma_query_response(response: dict[str, Any], *, score_threshold: float) -> list[RetrievedChunk]:
    """Преобразует ответ Chroma ``query`` в список ``RetrievedChunk``."""

    documents = _first_result_list(response.get("documents"))
    metadatas = _first_result_list(response.get("metadatas"))
    distances = _first_result_list(response.get("distances"))
    chunks: list[RetrievedChunk] = []
    for index, (text, metadata, distance) in enumerate(zip(documents, metadatas, distances), start=1):
        metadata_dict = dict(metadata or {})
        doc_id = metadata_dict.get("doc_id")
        chunk_id = metadata_dict.get("chunk_id")
        if not isinstance(doc_id, int):
            raise ValueError(f"Chroma вернула результат без целочисленного metadata.doc_id: {metadata_dict!r}.")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"Chroma вернула результат без непустого metadata.chunk_id: {metadata_dict!r}.")
        distance_value = float(distance) if distance is not None else None
        score = 1.0 - distance_value if distance_value is not None else None
        if score is not None and score < score_threshold:
            continue
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=str(text),
                metadata=metadata_dict,
                distance=distance_value,
                score=score,
                rank=index,
            )
        )
    return chunks


def _first_result_list(value: Any) -> list[Any]:
    """Возвращает первый список результатов из batch-ответа Chroma."""

    if value is None:
        return []
    if not isinstance(value, list) or not value:
        return []
    first = value[0]
    if first is None:
        return []
    if not isinstance(first, list):
        raise ValueError(f"Ожидался batch-список результатов Chroma, получено: {type(first)!r}.")
    return first


__all__ = [
    "RetrievedChunk",
    "RetrievalResult",
    "dense_retrieve",
    "evaluate_dense_retrieval",
]