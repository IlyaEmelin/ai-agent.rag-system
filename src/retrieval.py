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
    where: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    """Выполняет dense retrieval одного запроса через Chroma.

    Аргументы:
        collection: Chroma collection с чанками и embeddings.
        embedding_model: Wrapper из ``src.embeddings`` с методом ``embed_query``.
        query: Пользовательский вопрос.
        settings: Блок ``settings.retrieval``.
        top_k: Необязательное переопределение числа возвращаемых чанков.
        where: Необязательный Chroma-фильтр по metadata.

    Возвращает:
        Список найденных чанков с metadata, distance и rank.
    """

    requested_top_k = int(top_k if top_k is not None else settings.top_k)
    if requested_top_k <= 0:
        raise ValueError(f"top_k должен быть положительным числом, получено: {requested_top_k}.")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"Текст запроса должен быть непустой строкой, получено: {query!r}.")

    fetch_k = max(requested_top_k, int(settings.fetch_k))
    query_embedding = embedding_model.embed_query(query)
    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": fetch_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where
    response = collection.query(**query_kwargs)
    chunks = _parse_chroma_query_response(response, score_threshold=settings.score_threshold)
    return chunks[:requested_top_k]


def evaluate_dense_retrieval(
    collection: Any,
    embedding_model: Any,
    queries: Sequence[EvaluationQuery],
    settings: RetrievalSettings,
    *,
    top_k: int | None = None,
    include_gap_queries: bool = False,
    where: dict[str, Any] | None = None,
) -> list[RetrievalResult]:
    """Прогоняет dense retrieval по набору тестовых запросов.

    Аргументы:
        collection: Chroma collection.
        embedding_model: Wrapper embedding-модели.
        queries: Тестовые запросы датасета.
        settings: Настройки retrieval.
        top_k: Необязательное число возвращаемых чанков.
        include_gap_queries: Нужно ли включать ``gap``-запросы в прогон.
        where: Необязательный Chroma-фильтр по metadata.

    Возвращает:
        Список ``RetrievalResult``.
    """

    results: list[RetrievalResult] = []
    for evaluation_query in queries:
        if evaluation_query.kind == "gap" and not include_gap_queries:
            continue
        chunks = dense_retrieve(collection, embedding_model, evaluation_query.query, settings, top_k=top_k, where=where)
        results.append(
            RetrievalResult(
                query=evaluation_query.query,
                kind=evaluation_query.kind,
                relevant_docs=list(evaluation_query.relevant_docs),
                retrieved_chunks=chunks,
            )
        )
    return results


def format_retrieved_chunks(chunks: Sequence[RetrievedChunk], *, text_preview_chars: int = 220) -> list[dict[str, Any]]:
    """Форматирует найденные чанки для человекочитаемого вывода в notebook.

    Аргументы:
        chunks: Результаты retrieval.
        text_preview_chars: Максимальная длина текстового превью.

    Возвращает:
        Список строк-словарей с основными metadata и score.
    """

    if text_preview_chars <= 0:
        raise ValueError(f"text_preview_chars должен быть положительным числом, получено: {text_preview_chars}.")
    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        rows.append(
            {
                "rank": chunk.rank,
                "doc_id": chunk.doc_id,
                "title": chunk.metadata.get("title", ""),
                "category": chunk.metadata.get("category", ""),
                "chunk_id": chunk.chunk_id,
                "score": round(chunk.score, 6) if chunk.score is not None else None,
                "distance": round(chunk.distance, 6) if chunk.distance is not None else None,
                "text_preview": _trim_text(chunk.text, text_preview_chars),
            }
        )
    return rows


def summarize_score_distribution(results: Sequence[RetrievalResult]) -> list[dict[str, int | float | str | None]]:
    """Собирает распределение top-1 score отдельно для answerable и gap-запросов.

    Аргументы:
        results: Результаты retrieval, желательно с ``include_gap_queries=True``.

    Возвращает:
        Таблица статистик top-1 score по группам ``answerable`` и ``gap``.
    """

    groups: dict[str, list[float]] = {"answerable": [], "gap": []}
    for result in results:
        group_name = "gap" if result.kind == "gap" else "answerable"
        if result.retrieved_chunks and result.retrieved_chunks[0].score is not None:
            groups[group_name].append(float(result.retrieved_chunks[0].score))

    rows: list[dict[str, int | float | str | None]] = []
    for group_name, scores in groups.items():
        sorted_scores = sorted(scores)
        rows.append(
            {
                "group": group_name,
                "queries_with_results": len(sorted_scores),
                "min_top1_score": round(sorted_scores[0], 6) if sorted_scores else None,
                "p25_top1_score": round(_quantile(sorted_scores, 0.25), 6) if sorted_scores else None,
                "median_top1_score": round(_quantile(sorted_scores, 0.50), 6) if sorted_scores else None,
                "p75_top1_score": round(_quantile(sorted_scores, 0.75), 6) if sorted_scores else None,
                "max_top1_score": round(sorted_scores[-1], 6) if sorted_scores else None,
            }
        )
    return rows


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


def _trim_text(text: str, max_chars: int) -> str:
    """Обрезает текст до заданной длины для диагностических таблиц."""

    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 1]}…"


def _quantile(values: Sequence[float], q: float) -> float:
    """Возвращает линейно интерполированный квантиль для непустого списка."""

    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    bounded_q = min(max(float(q), 0.0), 1.0)
    position = (len(values) - 1) * bounded_q
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(values) - 1)
    fraction = position - lower_index
    return float(values[lower_index] * (1.0 - fraction) + values[upper_index] * fraction)


__all__ = [
    "RetrievedChunk",
    "RetrievalResult",
    "dense_retrieve",
    "evaluate_dense_retrieval",
    "format_retrieved_chunks",
    "summarize_score_distribution",
]