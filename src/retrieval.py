"""Утилиты retrieval-пайплайна для dense, sparse и hybrid поиска."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence

from langchain_core.documents import Document

from src.config import RetrievalSettings
from src.data_loading import EvaluationQuery

TOKEN_PATTERN = re.compile(r"[0-9a-zа-яё]+", flags=re.IGNORECASE)


@dataclass(frozen=True)
class RetrievedChunk:
    """Один найденный retrieval-результат.

    Атрибуты:
        chunk_id: Идентификатор чанка из metadata.
        doc_id: Идентификатор исходного документа.
        text: Текст найденного чанка.
        metadata: Metadata найденного чанка.
        distance: Расстояние Chroma до запроса, если оно возвращено.
        score: Основной score результата: similarity score для dense, BM25-score
            для BM25 или RRF-score для hybrid retrieval.
        rank: Позиция результата в выдаче, начиная с 1.
        retrieval_method: Имя retrieval-метода, сформировавшего результат.
        source_scores: Scores исходных retrieval-методов для диагностики.
        source_ranks: Ранги исходных retrieval-методов для диагностики.
    """

    chunk_id: str
    doc_id: int
    text: str
    metadata: dict[str, Any]
    distance: float | None
    score: float | None
    rank: int
    retrieval_method: str = "dense"
    source_scores: dict[str, float] | None = None
    source_ranks: dict[str, int] | None = None


@dataclass(frozen=True)
class RetrievalResult:
    """Результат retrieval для одного тестового запроса.

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


@dataclass(frozen=True)
class BM25Index:
    """BM25-индекс по тем же чанкам, что используются в dense retrieval.

    Атрибуты:
        chunks: Чанки LangChain ``Document``.
        tokenized_corpus: Токенизированные тексты чанков.
        model: Объект ``rank_bm25.BM25Okapi``.
    """

    chunks: list[Document]
    tokenized_corpus: list[list[str]]
    model: Any


def tokenize_for_bm25(text: str) -> list[str]:
    """Токенизирует русский и латинский текст для BM25.

    Токенизация намеренно простая и воспроизводимая: нижний регистр и выделение
    буквенно-цифровых последовательностей. Такой вариант хорошо ловит точные
    термины, названия настроек, коды ошибок и числа, дополняя dense retrieval.

    Аргументы:
        text: Исходный текст.

    Возвращает:
        Список токенов в нижнем регистре.
    """

    if not isinstance(text, str):
        raise ValueError(f"Текст для BM25 должен быть строкой, получено: {type(text)!r}.")
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def build_bm25_index(chunks: Sequence[Document]) -> BM25Index:
    """Строит BM25-индекс по списку чанков.

    Аргументы:
        chunks: Чанки документов с ``page_content`` и metadata.

    Возвращает:
        ``BM25Index`` с исходными чанками, токенами и моделью BM25.

    Исключения:
        ImportError: Если зависимость ``rank-bm25`` не установлена.
        ValueError: Если список чанков пуст или metadata некорректны.
    """

    if not chunks:
        raise ValueError("Список чанков для BM25 не должен быть пустым.")
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError("Пакет rank-bm25 не установлен. Установите зависимости: pip install -r requirements.txt") from exc

    normalized_chunks = list(chunks)
    tokenized_corpus: list[list[str]] = []
    for index, chunk in enumerate(normalized_chunks):
        _validate_document_chunk(chunk, index=index)
        tokens = tokenize_for_bm25(chunk.page_content)
        tokenized_corpus.append(tokens if tokens else [""])
    return BM25Index(chunks=normalized_chunks, tokenized_corpus=tokenized_corpus, model=BM25Okapi(tokenized_corpus))


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

    requested_top_k = _resolve_top_k(settings, top_k)
    _validate_query(query)

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
    return _rerank_chunks(chunks[:requested_top_k], retrieval_method="dense")


def bm25_retrieve(
    index: BM25Index,
    query: str,
    settings: RetrievalSettings,
    *,
    top_k: int | None = None,
    where: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    """Выполняет BM25 retrieval одного запроса.

    Аргументы:
        index: BM25-индекс по чанкам.
        query: Пользовательский вопрос.
        settings: Блок ``settings.retrieval``.
        top_k: Необязательное переопределение числа возвращаемых чанков.
        where: Необязательный простой фильтр metadata по равенству.

    Возвращает:
        Список найденных чанков с BM25-score и metadata.
    """

    requested_top_k = _resolve_top_k(settings, top_k)
    _validate_query(query)
    query_tokens = tokenize_for_bm25(query)
    scores = index.model.get_scores(query_tokens if query_tokens else [""])
    candidate_rows: list[tuple[int, float]] = []
    for chunk_index, score in enumerate(scores):
        document = index.chunks[chunk_index]
        if where and not _metadata_matches_where(document.metadata, where):
            continue
        candidate_rows.append((chunk_index, float(score)))
    candidate_rows.sort(key=lambda item: (-item[1], item[0]))

    chunks: list[RetrievedChunk] = []
    fetch_k = max(requested_top_k, int(settings.fetch_k))
    for rank, (chunk_index, score) in enumerate(candidate_rows[:fetch_k], start=1):
        chunks.append(_document_to_retrieved_chunk(index.chunks[chunk_index], rank=rank, score=score, retrieval_method="bm25"))
    return chunks[:requested_top_k]


def reciprocal_rank_fusion(
    ranked_results: Sequence[Sequence[RetrievedChunk]],
    settings: RetrievalSettings,
    *,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Объединяет несколько ранжированных списков через Reciprocal Rank Fusion.

    Аргументы:
        ranked_results: Последовательность ranked lists от разных retrieval-методов.
        settings: Блок ``settings.retrieval`` с параметром ``rrf_k``.
        top_k: Необязательное число итоговых результатов.

    Возвращает:
        Единый ранжированный список без дублей ``chunk_id``.
    """

    requested_top_k = _resolve_top_k(settings, top_k)
    rrf_k = int(settings.rrf_k)
    if rrf_k <= 0:
        raise ValueError(f"RRF_K должен быть положительным числом, получено: {rrf_k}.")

    fused: dict[str, dict[str, Any]] = {}
    for result_list in ranked_results:
        for chunk in result_list:
            if chunk.chunk_id not in fused:
                fused[chunk.chunk_id] = {
                    "chunk": chunk,
                    "score": 0.0,
                    "source_scores": {},
                    "source_ranks": {},
                }
            method = chunk.retrieval_method or "unknown"
            source_scores = fused[chunk.chunk_id]["source_scores"]
            source_ranks = fused[chunk.chunk_id]["source_ranks"]
            source_scores[method] = float(chunk.score) if chunk.score is not None else 0.0
            source_ranks[method] = int(chunk.rank)
            fused[chunk.chunk_id]["score"] += 1.0 / (rrf_k + int(chunk.rank))

    ranked_items = sorted(
        fused.values(),
        key=lambda item: (-float(item["score"]), min(item["source_ranks"].values()), item["chunk"].chunk_id),
    )

    fused_chunks: list[RetrievedChunk] = []
    for rank, item in enumerate(ranked_items[:requested_top_k], start=1):
        base_chunk: RetrievedChunk = item["chunk"]
        fused_chunks.append(
            RetrievedChunk(
                chunk_id=base_chunk.chunk_id,
                doc_id=base_chunk.doc_id,
                text=base_chunk.text,
                metadata=base_chunk.metadata,
                distance=None,
                score=float(item["score"]),
                rank=rank,
                retrieval_method="hybrid_rrf",
                source_scores=dict(item["source_scores"]),
                source_ranks=dict(item["source_ranks"]),
            )
        )
    return fused_chunks


def hybrid_retrieve(
    collection: Any,
    embedding_model: Any,
    bm25_index: BM25Index,
    query: str,
    settings: RetrievalSettings,
    *,
    top_k: int | None = None,
    where: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    """Выполняет hybrid retrieval: dense + BM25 + RRF.

    Аргументы:
        collection: Chroma collection для dense retrieval.
        embedding_model: Wrapper embedding-модели.
        bm25_index: BM25-индекс по тем же чанкам.
        query: Пользовательский вопрос.
        settings: Блок ``settings.retrieval``.
        top_k: Необязательное число итоговых результатов.
        where: Необязательный metadata-фильтр.

    Возвращает:
        Единый ранжированный список чанков. Если BM25 или RRF выключены в
        настройках, функция безопасно возвращает dense-only результат.
    """

    requested_top_k = _resolve_top_k(settings, top_k)
    candidate_k = max(requested_top_k, int(settings.fetch_k))
    dense_chunks = dense_retrieve(collection, embedding_model, query, settings, top_k=candidate_k, where=where)
    if not settings.bm25_enabled:
        return _rerank_chunks(dense_chunks[:requested_top_k], retrieval_method="dense")

    bm25_chunks = bm25_retrieve(bm25_index, query, settings, top_k=candidate_k, where=where)
    if not settings.rrf_enabled:
        return _merge_ranked_chunks_without_rrf([dense_chunks, bm25_chunks], top_k=requested_top_k)
    return reciprocal_rank_fusion([dense_chunks, bm25_chunks], settings, top_k=requested_top_k)


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
    """Прогоняет dense retrieval по набору тестовых запросов."""

    results: list[RetrievalResult] = []
    for evaluation_query in queries:
        if evaluation_query.kind == "gap" and not include_gap_queries:
            continue
        chunks = dense_retrieve(collection, embedding_model, evaluation_query.query, settings, top_k=top_k, where=where)
        results.append(_build_retrieval_result(evaluation_query, chunks))
    return results


def evaluate_bm25_retrieval(
    index: BM25Index,
    queries: Sequence[EvaluationQuery],
    settings: RetrievalSettings,
    *,
    top_k: int | None = None,
    include_gap_queries: bool = False,
    where: dict[str, Any] | None = None,
) -> list[RetrievalResult]:
    """Прогоняет BM25 retrieval по набору тестовых запросов.

    Аргументы:
        index: BM25-индекс по чанкам.
        queries: Тестовые запросы датасета.
        settings: Настройки retrieval.
        top_k: Необязательное число возвращаемых чанков.
        include_gap_queries: Нужно ли включать ``gap``-запросы в прогон.
        where: Необязательный metadata-фильтр.

    Возвращает:
        Список ``RetrievalResult``.
    """

    results: list[RetrievalResult] = []
    for evaluation_query in queries:
        if evaluation_query.kind == "gap" and not include_gap_queries:
            continue
        chunks = bm25_retrieve(index, evaluation_query.query, settings, top_k=top_k, where=where)
        results.append(_build_retrieval_result(evaluation_query, chunks))
    return results


def evaluate_hybrid_retrieval(
    collection: Any,
    embedding_model: Any,
    bm25_index: BM25Index,
    queries: Sequence[EvaluationQuery],
    settings: RetrievalSettings,
    *,
    top_k: int | None = None,
    include_gap_queries: bool = False,
    where: dict[str, Any] | None = None,
) -> list[RetrievalResult]:
    """Прогоняет hybrid retrieval по набору тестовых запросов."""

    results: list[RetrievalResult] = []
    for evaluation_query in queries:
        if evaluation_query.kind == "gap" and not include_gap_queries:
            continue
        chunks = hybrid_retrieve(
            collection,
            embedding_model,
            bm25_index,
            evaluation_query.query,
            settings,
            top_k=top_k,
            where=where,
        )
        results.append(_build_retrieval_result(evaluation_query, chunks))
    return results


def format_retrieved_chunks(chunks: Sequence[RetrievedChunk], *, text_preview_chars: int = 220) -> list[dict[str, Any]]:
    """Форматирует найденные чанки для человекочитаемого вывода в notebook."""

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
                "method": chunk.retrieval_method,
                "score": round(chunk.score, 6) if chunk.score is not None else None,
                "distance": round(chunk.distance, 6) if chunk.distance is not None else None,
                "source_scores": _round_score_dict(chunk.source_scores),
                "source_ranks": dict(chunk.source_ranks or {}),
                "text_preview": _trim_text(chunk.text, text_preview_chars),
            }
        )
    return rows


def summarize_score_distribution(results: Sequence[RetrievalResult]) -> list[dict[str, int | float | str | None]]:
    """Собирает распределение top-1 score отдельно для answerable и gap-запросов."""

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


def _build_retrieval_result(evaluation_query: EvaluationQuery, chunks: Sequence[RetrievedChunk]) -> RetrievalResult:
    """Собирает ``RetrievalResult`` из запроса и найденных чанков."""

    return RetrievalResult(
        query=evaluation_query.query,
        kind=evaluation_query.kind,
        relevant_docs=list(evaluation_query.relevant_docs),
        retrieved_chunks=list(chunks),
    )


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
                retrieval_method="dense",
                source_scores={"dense": score} if score is not None else {},
                source_ranks={"dense": index},
            )
        )
    return chunks


def _document_to_retrieved_chunk(document: Document, *, rank: int, score: float, retrieval_method: str) -> RetrievedChunk:
    """Преобразует LangChain ``Document`` в ``RetrievedChunk``."""

    metadata = dict(document.metadata)
    doc_id = metadata.get("doc_id")
    chunk_id = metadata.get("chunk_id")
    if not isinstance(doc_id, int):
        raise ValueError(f"У чанка отсутствует целочисленный metadata.doc_id: {metadata!r}.")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError(f"У чанка отсутствует непустой metadata.chunk_id: {metadata!r}.")
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=document.page_content,
        metadata=metadata,
        distance=None,
        score=float(score),
        rank=rank,
        retrieval_method=retrieval_method,
        source_scores={retrieval_method: float(score)},
        source_ranks={retrieval_method: rank},
    )


def _merge_ranked_chunks_without_rrf(ranked_results: Sequence[Sequence[RetrievedChunk]], *, top_k: int) -> list[RetrievedChunk]:
    """Объединяет выдачи без RRF, сохраняя первый встреченный чанк."""

    merged: list[RetrievedChunk] = []
    seen: set[str] = set()
    for result_list in ranked_results:
        for chunk in result_list:
            if chunk.chunk_id in seen:
                continue
            merged.append(chunk)
            seen.add(chunk.chunk_id)
            if len(merged) >= top_k:
                return _rerank_chunks(merged, retrieval_method="hybrid_concat")
    return _rerank_chunks(merged, retrieval_method="hybrid_concat")


def _rerank_chunks(chunks: Sequence[RetrievedChunk], *, retrieval_method: str | None = None) -> list[RetrievedChunk]:
    """Перенумеровывает ранги выдачи после фильтрации или слияния."""

    reranked: list[RetrievedChunk] = []
    for rank, chunk in enumerate(chunks, start=1):
        method = retrieval_method or chunk.retrieval_method
        source_scores = dict(chunk.source_scores or {})
        source_ranks = dict(chunk.source_ranks or {})
        if method not in source_ranks:
            source_ranks[method] = rank
        if method not in source_scores and chunk.score is not None:
            source_scores[method] = float(chunk.score)
        reranked.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                metadata=chunk.metadata,
                distance=chunk.distance,
                score=chunk.score,
                rank=rank,
                retrieval_method=method,
                source_scores=source_scores,
                source_ranks=source_ranks,
            )
        )
    return reranked


def _metadata_matches_where(metadata: dict[str, Any], where: dict[str, Any]) -> bool:
    """Проверяет простой metadata-фильтр по равенству значений."""

    for key, expected in where.items():
        if isinstance(expected, dict):
            if "$eq" in expected and metadata.get(key) != expected["$eq"]:
                return False
            if "$in" in expected and metadata.get(key) not in expected["$in"]:
                return False
        elif metadata.get(key) != expected:
            return False
    return True


def _validate_document_chunk(document: Document, *, index: int) -> None:
    """Проверяет чанк перед добавлением в BM25-индекс."""

    if not isinstance(document.page_content, str) or not document.page_content.strip():
        raise ValueError(f"Чанк index={index} имеет пустой page_content.")
    metadata = dict(document.metadata or {})
    if not isinstance(metadata.get("doc_id"), int):
        raise ValueError(f"Чанк index={index} не содержит целочисленный metadata.doc_id: {metadata!r}.")
    if not isinstance(metadata.get("chunk_id"), str) or not metadata.get("chunk_id"):
        raise ValueError(f"Чанк index={index} не содержит непустой metadata.chunk_id: {metadata!r}.")


def _resolve_top_k(settings: RetrievalSettings, top_k: int | None) -> int:
    """Определяет итоговое значение top-K."""

    requested_top_k = int(top_k if top_k is not None else settings.top_k)
    if requested_top_k <= 0:
        raise ValueError(f"top_k должен быть положительным числом, получено: {requested_top_k}.")
    return requested_top_k


def _validate_query(query: str) -> None:
    """Проверяет пользовательский retrieval-запрос."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"Текст запроса должен быть непустой строкой, получено: {query!r}.")


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


def _round_score_dict(values: dict[str, float] | None) -> dict[str, float]:
    """Округляет словарь score-значений для компактного вывода."""

    return {key: round(float(value), 6) for key, value in (values or {}).items()}


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
    "BM25Index",
    "RetrievedChunk",
    "RetrievalResult",
    "bm25_retrieve",
    "build_bm25_index",
    "dense_retrieve",
    "evaluate_bm25_retrieval",
    "evaluate_dense_retrieval",
    "evaluate_hybrid_retrieval",
    "format_retrieved_chunks",
    "hybrid_retrieve",
    "reciprocal_rank_fusion",
    "summarize_score_distribution",
    "tokenize_for_bm25",
]
