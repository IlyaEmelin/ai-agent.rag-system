"""Утилиты расчёта retrieval- и generation-метрик."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.retrieval import RetrievalResult


DEFAULT_RECALL_K_VALUES = (1, 3, 5, 10, 20)


@dataclass(frozen=True)
class RetrievalMetricsSummary:
    """Агрегированная сводка retrieval-метрик.

    Атрибуты:
        queries: Количество оценённых запросов.
        precision_at_k: Средний ``precision@k``.
        recall_at_k: Средний ``recall@k``.
        mrr: Mean Reciprocal Rank по уникальным ``doc_id``.
        recall_curve: Средний ``recall@K`` для набора K.
    """

    queries: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    recall_curve: dict[int, float]

    def as_dict(self) -> dict[str, int | float | dict[int, float]]:
        """Возвращает метрики в виде словаря для notebook-таблиц."""

        return {
            "queries": self.queries,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "recall_curve": self.recall_curve,
        }


def compute_retrieval_metrics(
    results: Sequence[RetrievalResult],
    *,
    k: int = 5,
    recall_k_values: Sequence[int] = DEFAULT_RECALL_K_VALUES,
) -> RetrievalMetricsSummary:
    """Считает агрегированные retrieval-метрики по результатам поиска.

    ``gap``-запросы и запросы без ``relevant_docs`` исключаются на этом уровне,
    чтобы метрики оценивали только вопросы с ответом в базе знаний.

    Аргументы:
        results: Результаты retrieval.
        k: Значение K для ``precision@K`` и ``recall@K``.
        recall_k_values: Набор K для recall-кривой.

    Возвращает:
        ``RetrievalMetricsSummary``.
    """

    if k <= 0:
        raise ValueError(f"k должен быть положительным числом, получено: {k}.")
    normalized_recall_k_values = tuple(sorted({int(value) for value in recall_k_values if int(value) > 0}))
    evaluated = [result for result in results if result.kind != "gap" and result.relevant_docs]
    if not evaluated:
        return RetrievalMetricsSummary(queries=0, precision_at_k=0.0, recall_at_k=0.0, mrr=0.0, recall_curve={})

    precision_values = [precision_at_k(result.unique_retrieved_doc_ids, result.relevant_docs, k) for result in evaluated]
    recall_values = [recall_at_k(result.unique_retrieved_doc_ids, result.relevant_docs, k) for result in evaluated]
    mrr_values = [reciprocal_rank(result.unique_retrieved_doc_ids, result.relevant_docs) for result in evaluated]
    recall_curve = {
        value: round(_mean(recall_at_k(result.unique_retrieved_doc_ids, result.relevant_docs, value) for result in evaluated), 6)
        for value in normalized_recall_k_values
    }
    return RetrievalMetricsSummary(
        queries=len(evaluated),
        precision_at_k=round(_mean(precision_values), 6),
        recall_at_k=round(_mean(recall_values), 6),
        mrr=round(_mean(mrr_values), 6),
        recall_curve=recall_curve,
    )


def compute_retrieval_metrics_by_kind(
    results: Sequence[RetrievalResult],
    *,
    k: int = 5,
    recall_k_values: Sequence[int] = DEFAULT_RECALL_K_VALUES,
) -> dict[str, RetrievalMetricsSummary]:
    """Считает retrieval-метрики отдельно по типам answerable-запросов.

    ``gap``-запросы не включаются: для них качество оценивается на этапе
    генерации и политики отказа, а не через retrieval-recall.

    Аргументы:
        results: Результаты retrieval.
        k: Значение K для ``precision@K`` и ``recall@K``.
        recall_k_values: Набор K для recall-кривой.

    Возвращает:
        Словарь ``kind -> RetrievalMetricsSummary``.
    """

    grouped: dict[str, list[RetrievalResult]] = {}
    for result in results:
        if result.kind == "gap":
            continue
        grouped.setdefault(result.kind, []).append(result)
    return {
        kind: compute_retrieval_metrics(kind_results, k=k, recall_k_values=recall_k_values)
        for kind, kind_results in sorted(grouped.items())
    }


def build_weak_query_diagnostics(
    results: Sequence[RetrievalResult],
    *,
    k: int = 5,
    extended_k_values: Sequence[int] = DEFAULT_RECALL_K_VALUES,
    limit: int | None = None,
) -> list[dict[str, object]]:
    """Строит диагностическую таблицу слабых retrieval-запросов.

    В таблицу попадают answerable-запросы с ``recall@k = 0`` или низким MRR.
    Для каждого запроса дополнительно показывается, появляется ли релевантный
    документ глубже в выдаче, например на ``K=10`` или ``K=20``.

    Аргументы:
        results: Результаты retrieval.
        k: Основная глубина диагностики, обычно ``5``.
        extended_k_values: Набор K для дополнительных recall-признаков.
        limit: Необязательный лимит числа строк после сортировки.

    Возвращает:
        Список строк-словарей для notebook-таблицы.
    """

    if k <= 0:
        raise ValueError(f"k должен быть положительным числом, получено: {k}.")
    normalized_extended_k = tuple(sorted({int(value) for value in extended_k_values if int(value) > 0}))
    rows: list[dict[str, object]] = []
    for result in results:
        if result.kind == "gap" or not result.relevant_docs:
            continue
        unique_doc_ids = result.unique_retrieved_doc_ids
        query_recall = recall_at_k(unique_doc_ids, result.relevant_docs, k)
        query_rr = reciprocal_rank(unique_doc_ids, result.relevant_docs)
        first_rank = first_relevant_rank(unique_doc_ids, result.relevant_docs)
        if query_recall > 0.0 and query_rr >= 1.0 / max(k, 1):
            continue
        recall_columns = {
            f"recall@{value}": round(recall_at_k(unique_doc_ids, result.relevant_docs, value), 6)
            for value in normalized_extended_k
        }
        rows.append(
            {
                "query": result.query,
                "kind": result.kind,
                "relevant_docs": list(result.relevant_docs),
                "top_doc_ids": unique_doc_ids[: max(normalized_extended_k, default=k)],
                "first_relevant_rank": first_rank,
                "reciprocal_rank": round(query_rr, 6),
                "diagnosis": classify_retrieval_issue(first_rank, k=k, max_k=max(normalized_extended_k, default=k)),
                **recall_columns,
            }
        )
    rows.sort(key=lambda row: (row["reciprocal_rank"], row["first_relevant_rank"] is None, row["first_relevant_rank"] or 10**9))
    return rows[:limit] if limit is not None else rows


def first_relevant_rank(retrieved_doc_ids: Sequence[int], relevant_doc_ids: Sequence[int]) -> int | None:
    """Возвращает ранг первого релевантного документа или ``None``.

    Аргументы:
        retrieved_doc_ids: Найденные ``doc_id`` в порядке ранжирования.
        relevant_doc_ids: Эталонные релевантные ``doc_id``.

    Возвращает:
        Позиция первого релевантного документа, начиная с ``1``.
    """

    relevant = set(relevant_doc_ids)
    if not relevant:
        return None
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant:
            return rank
    return None


def classify_retrieval_issue(first_rank: int | None, *, k: int = 5, max_k: int = 20) -> str:
    """Классифицирует слабый retrieval-пример по рангу релевантного документа.

    Аргументы:
        first_rank: Ранг первого релевантного документа или ``None``.
        k: Целевая глубина контекста.
        max_k: Максимальная глубина просмотренной выдачи.

    Возвращает:
        Короткая диагностическая категория.
    """

    if first_rank is None:
        return f"релевантный документ не найден в top-{max_k}"
    if first_rank <= k:
        return "релевантный документ найден в целевом top-K, но MRR низкий"
    if first_rank <= max_k:
        return "релевантный документ найден, но стоит слишком низко"
    return f"релевантный документ глубже top-{max_k}"


def precision_at_k(retrieved_doc_ids: Sequence[int], relevant_doc_ids: Sequence[int], k: int) -> float:
    """Считает ``precision@K`` по уникальным ``doc_id``.

    Аргументы:
        retrieved_doc_ids: Найденные ``doc_id`` в порядке ранжирования.
        relevant_doc_ids: Эталонные релевантные ``doc_id``.
        k: Глубина выдачи.

    Возвращает:
        Долю найденных на первых K позиций документов, которые релевантны.
    """

    if k <= 0:
        raise ValueError(f"k должен быть положительным числом, получено: {k}.")
    if not retrieved_doc_ids:
        return 0.0
    relevant = set(relevant_doc_ids)
    if not relevant:
        return 0.0
    top_ids = list(retrieved_doc_ids)[:k]
    return len([doc_id for doc_id in top_ids if doc_id in relevant]) / k


def recall_at_k(retrieved_doc_ids: Sequence[int], relevant_doc_ids: Sequence[int], k: int) -> float:
    """Считает ``recall@K`` по уникальным ``doc_id``.

    Аргументы:
        retrieved_doc_ids: Найденные ``doc_id`` в порядке ранжирования.
        relevant_doc_ids: Эталонные релевантные ``doc_id``.
        k: Глубина выдачи.

    Возвращает:
        Долю эталонных документов, найденных на первых K позициях.
    """

    if k <= 0:
        raise ValueError(f"k должен быть положительным числом, получено: {k}.")
    relevant = set(relevant_doc_ids)
    if not relevant:
        return 0.0
    top_ids = set(list(retrieved_doc_ids)[:k])
    return len(top_ids.intersection(relevant)) / len(relevant)


def reciprocal_rank(retrieved_doc_ids: Sequence[int], relevant_doc_ids: Sequence[int]) -> float:
    """Считает reciprocal rank первого релевантного документа.

    Аргументы:
        retrieved_doc_ids: Найденные ``doc_id`` в порядке ранжирования.
        relevant_doc_ids: Эталонные релевантные ``doc_id``.

    Возвращает:
        ``1 / rank`` первого релевантного документа или ``0.0``.
    """

    relevant = set(relevant_doc_ids)
    if not relevant:
        return 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def find_zero_recall_queries(results: Sequence[RetrievalResult], *, k: int = 5, limit: int = 5) -> list[RetrievalResult]:
    """Возвращает примеры запросов с нулевым ``recall@K``.

    Аргументы:
        results: Результаты retrieval.
        k: Глубина выдачи для recall.
        limit: Максимальное число примеров.

    Возвращает:
        Список результатов с ``recall@K = 0``.
    """

    weak_results = [
        result
        for result in results
        if result.kind != "gap" and result.relevant_docs and recall_at_k(result.unique_retrieved_doc_ids, result.relevant_docs, k) == 0.0
    ]
    return weak_results[:limit]


def _mean(values: Sequence[float]) -> float:
    """Возвращает среднее значение или ``0.0`` для пустой последовательности."""

    value_list = list(values)
    if not value_list:
        return 0.0
    return sum(value_list) / len(value_list)


__all__ = [
    "DEFAULT_RECALL_K_VALUES",
    "RetrievalMetricsSummary",
    "build_weak_query_diagnostics",
    "classify_retrieval_issue",
    "compute_retrieval_metrics",
    "compute_retrieval_metrics_by_kind",
    "first_relevant_rank",
    "find_zero_recall_queries",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]