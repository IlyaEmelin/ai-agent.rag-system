"""Утилиты LLM-as-judge и финальной диагностики качества RAG."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from src.config import GenerationSettings, JudgeSettings, Settings
from src.data_loading import EvaluationQuery, KnowledgeDocument
from src.generation import OpenAIChatClient, RagResponse, build_rag_context, generate_answer, run_rag_pipeline
from src.metrics import RetrievalMetricsSummary, compute_retrieval_metrics, compute_retrieval_metrics_by_kind, recall_at_k
from src.retrieval import BM25Index, RetrievedChunk, RetrievalResult


JUDGE_SCHEMA_VERSION = "1.0"
JUDGE_SCORE_FIELDS = ("faithfulness", "answer_relevance", "correctness")
DEFAULT_JUDGE_RETRY_COUNT = 1
DEFAULT_INTERMEDIATE_SAVE_EVERY = 5
DEFAULT_WORST_CASE_LIMIT = 8
DEFAULT_CORRECTNESS_TARGET = 0.90
DEFAULT_RECALL_AT_5_TARGET = 0.85


@dataclass(frozen=True)
class JudgeResult:
    """Структурированный результат оценки одного ответа LLM-судьёй.

    Атрибуты:
        faithfulness: Бинарная оценка опоры ответа на контекст.
        answer_relevance: Бинарная оценка релевантности ответа вопросу.
        correctness: Бинарная сверка ответа с ``gold``.
        comment: Краткий комментарий судьи.
        raw_response: Исходный текст ответа судьи.
        parse_error: Ошибка парсинга или валидации, если она была.
        attempts: Количество попыток вызова судьи.
    """

    faithfulness: int
    answer_relevance: int
    correctness: int
    comment: str
    raw_response: str
    parse_error: str | None = None
    attempts: int = 1

    @property
    def is_valid(self) -> bool:
        """Проверяет, что judge-ответ был распарсен без ошибок."""

        return self.parse_error is None

    def as_dict(self) -> dict[str, Any]:
        """Возвращает результат оценки в виде JSON-совместимого словаря."""

        return {
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "correctness": self.correctness,
            "comment": self.comment,
            "raw_response": self.raw_response,
            "parse_error": self.parse_error,
            "attempts": self.attempts,
        }


@dataclass(frozen=True)
class EvaluationRecord:
    """Результат полного оценочного прогона одного тестового запроса.

    Атрибуты:
        query: Текст вопроса.
        kind: Тип запроса из датасета.
        relevant_docs: Эталонные релевантные документы.
        gold: Эталонный факт или эталон отказа.
        retrieved_chunks: Найденные чанки, переданные генератору.
        retrieved_doc_ids: Найденные ``doc_id`` с сохранением рангов.
        retrieval_scores: Scores и distances найденных чанков.
        answer: Ответ генератора.
        judge_result: Результат LLM-as-judge, если вызов судьи удался.
        error: Ошибка pipeline или judge-вызова, если она была.
        diagnostics: Дополнительная диагностика RAG-пайплайна.
    """

    query: str
    kind: str
    relevant_docs: list[int]
    gold: str
    retrieved_chunks: list[dict[str, Any]]
    retrieved_doc_ids: list[int]
    retrieval_scores: list[dict[str, Any]]
    answer: str
    judge_result: JudgeResult | None
    error: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Возвращает запись оценки в виде JSON-совместимого словаря."""

        return {
            "query": self.query,
            "kind": self.kind,
            "relevant_docs": list(self.relevant_docs),
            "gold": self.gold,
            "retrieved_chunks": list(self.retrieved_chunks),
            "retrieved_doc_ids": list(self.retrieved_doc_ids),
            "retrieval_scores": list(self.retrieval_scores),
            "answer": self.answer,
            "judge_result": self.judge_result.as_dict() if self.judge_result else None,
            "error": self.error,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class GenerationMetricsSummary:
    """Сводка generation-метрик по judge-результатам.

    Атрибуты:
        queries: Количество записей в сводке.
        evaluated: Количество записей с валидной оценкой судьи.
        faithfulness: Средняя бинарная faithfulness-оценка.
        answer_relevance: Средняя бинарная relevance-оценка.
        correctness: Средняя бинарная correctness-оценка.
        invalid_judge_results: Количество невалидных judge-ответов.
        pipeline_errors: Количество ошибок RAG-пайплайна.
        gap_hallucination_rate: Доля ``gap``-запросов без корректного отказа.
    """

    queries: int
    evaluated: int
    faithfulness: float
    answer_relevance: float
    correctness: float
    invalid_judge_results: int
    pipeline_errors: int
    gap_hallucination_rate: float

    def as_dict(self) -> dict[str, int | float]:
        """Возвращает сводку в виде словаря для notebook-таблиц."""

        return {
            "queries": self.queries,
            "evaluated": self.evaluated,
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "correctness": self.correctness,
            "invalid_judge_results": self.invalid_judge_results,
            "pipeline_errors": self.pipeline_errors,
            "gap_hallucination_rate": self.gap_hallucination_rate,
        }


@dataclass(frozen=True)
class QualityProfile:
    """Итоговый профиль качества RAG-системы.

    Атрибуты:
        retrieval: Общие retrieval-метрики.
        retrieval_by_kind: Retrieval-метрики по типам answerable-запросов.
        generation: Общие generation-метрики.
        generation_by_kind: Generation-метрики по всем типам запросов.
        targets: Проверка достижения целевых ориентиров.
    """

    retrieval: RetrievalMetricsSummary
    retrieval_by_kind: dict[str, RetrievalMetricsSummary]
    generation: GenerationMetricsSummary
    generation_by_kind: dict[str, GenerationMetricsSummary]
    targets: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Возвращает профиль качества в JSON-совместимом виде."""

        return {
            "retrieval": self.retrieval.as_dict(),
            "retrieval_by_kind": {kind: summary.as_dict() for kind, summary in self.retrieval_by_kind.items()},
            "generation": self.generation.as_dict(),
            "generation_by_kind": {kind: summary.as_dict() for kind, summary in self.generation_by_kind.items()},
            "targets": dict(self.targets),
        }


@dataclass(frozen=True)
class GoldenContextRecord:
    """Результат сравнения обычного RAG и golden-context режима."""

    query: str
    kind: str
    relevant_docs: list[int]
    gold: str
    rag_answer: str
    golden_answer: str
    rag_correctness: int | None
    golden_correctness: int | None
    delta: int | None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Возвращает запись golden-context сравнения как словарь."""

        return {
            "query": self.query,
            "kind": self.kind,
            "relevant_docs": list(self.relevant_docs),
            "gold": self.gold,
            "rag_answer": self.rag_answer,
            "golden_answer": self.golden_answer,
            "rag_correctness": self.rag_correctness,
            "golden_correctness": self.golden_correctness,
            "delta": self.delta,
            "error": self.error,
        }


class JudgeClient:
    """OpenAI-compatible клиент LLM-судьи с JSON-парсингом и retry."""

    def __init__(self, settings: JudgeSettings, *, retry_count: int = DEFAULT_JUDGE_RETRY_COUNT) -> None:
        """Создаёт клиента судьи по блоку ``settings.judge``.

        Аргументы:
            settings: Настройки judge-модели.
            retry_count: Количество повторных запросов при невалидном JSON.
        """

        validate_judge_settings(settings)
        if retry_count < 0:
            raise ValueError(f"retry_count не может быть отрицательным, получено: {retry_count}.")
        self.settings = settings
        self.retry_count = retry_count
        self._llm_client = OpenAIChatClient(_judge_settings_to_llm_settings(settings))

    def evaluate(self, *, query: str, context: str, answer: str, gold: str, kind: str) -> JudgeResult:
        """Оценивает один ответ и возвращает структурированный результат."""

        prompt = build_judge_prompt(query=query, context=context, answer=answer, gold=gold, kind=kind)
        last_raw = ""
        last_error = ""
        for attempt in range(1, self.retry_count + 2):
            try:
                response = self._llm_client.complete(prompt)
                last_raw = response.content
                return parse_judge_response(last_raw, attempts=attempt)
            except Exception as exc:  # noqa: BLE001 - нужно сохранить любую ошибку judge-вызова.
                last_error = str(exc)
        return build_failed_judge_result(last_raw, last_error or "Judge не вернул валидный JSON.", attempts=self.retry_count + 1)


def build_judge_prompt(*, query: str, context: str, answer: str, gold: str, kind: str) -> str:
    """Строит prompt для LLM-as-judge с требованием строгого JSON.

    Аргументы:
        query: Вопрос пользователя.
        context: Контекст, переданный генератору.
        answer: Ответ генератора.
        gold: Эталонный факт или эталон отказа.
        kind: Тип запроса: ``exact``, ``semantic``, ``multi`` или ``gap``.

    Возвращает:
        Prompt для judge-модели.
    """

    return f"""Ты независимый судья качества RAG-ответа службы поддержки онлайн-кинотеатра «Фильмоскоп».

Оцени ответ только по указанному контексту и эталону. Не добавляй факты от себя.

Метрики:
- faithfulness: 1, если ответ полностью следует из контекста или корректно отказывается из-за отсутствия ответа в контексте; иначе 0.
- answer_relevance: 1, если ответ прямо отвечает на вопрос пользователя или корректно отказывается для gap-запроса; иначе 0.
- correctness: 1, если ответ совпадает по смыслу с gold. Для kind="gap" корректным считается только согласованный отказ без неподтверждённых фактов. Иначе 0.

Верни только валидный JSON без markdown и без пояснений вне JSON.
Схема JSON:
{{
  "faithfulness": 0 или 1,
  "answer_relevance": 0 или 1,
  "correctness": 0 или 1,
  "comment": "краткое объяснение на русском"
}}

kind: {kind}
Вопрос:
{query}

Контекст:
{context or "<контекст пуст>"}

Ответ модели:
{answer or "<ответ пуст>"}

Gold:
{gold}
""".strip()


def parse_judge_response(raw_response: str, *, attempts: int = 1) -> JudgeResult:
    """Парсит и валидирует JSON-ответ LLM-судьи.

    Аргументы:
        raw_response: Текст ответа судьи.
        attempts: Номер успешной попытки.

    Возвращает:
        Валидный ``JudgeResult``.

    Исключения:
        ValueError: Если JSON не найден или схема невалидна.
    """

    raw_text = str(raw_response or "").strip()
    if not raw_text:
        raise ValueError("Judge вернул пустой ответ.")
    payload = json.loads(extract_json_object(raw_text))
    if not isinstance(payload, dict):
        raise ValueError("Judge JSON должен быть объектом.")
    scores = {field_name: _parse_binary_score(payload.get(field_name), field_name) for field_name in JUDGE_SCORE_FIELDS}
    comment = str(payload.get("comment", "")).strip()
    if not comment:
        comment = "Judge не вернул комментарий."
    return JudgeResult(
        faithfulness=scores["faithfulness"],
        answer_relevance=scores["answer_relevance"],
        correctness=scores["correctness"],
        comment=comment,
        raw_response=raw_text,
        parse_error=None,
        attempts=attempts,
    )


def safe_parse_judge_response(raw_response: str, *, attempts: int = 1) -> JudgeResult:
    """Безопасно парсит judge-ответ, возвращая нулевые оценки при ошибке."""

    try:
        return parse_judge_response(raw_response, attempts=attempts)
    except Exception as exc:  # noqa: BLE001 - fallback должен поймать любую ошибку парсинга.
        return build_failed_judge_result(raw_response, str(exc), attempts=attempts)


def extract_json_object(text: str) -> str:
    """Извлекает первый JSON-объект из текста ответа судьи."""

    stripped = str(text or "").strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1).strip()
    object_match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if object_match:
        return object_match.group(0).strip()
    raise ValueError("В ответе judge не найден JSON-объект.")


def build_failed_judge_result(raw_response: str, error: str, *, attempts: int = 1) -> JudgeResult:
    """Создаёт безопасный judge-result при ошибке вызова или парсинга."""

    return JudgeResult(
        faithfulness=0,
        answer_relevance=0,
        correctness=0,
        comment="Judge-оценка невалидна; запись требует ручной проверки.",
        raw_response=str(raw_response or ""),
        parse_error=str(error or "Неизвестная ошибка judge."),
        attempts=attempts,
    )


def run_evaluation_queries(
    queries: Sequence[EvaluationQuery],
    *,
    collection: Any,
    embedding_model: Any,
    bm25_index: BM25Index,
    settings: Settings,
    llm_client: OpenAIChatClient | None = None,
    judge_client: JudgeClient | None = None,
    top_k: int | None = None,
    output_path: str | Path | None = None,
    save_every: int = DEFAULT_INTERMEDIATE_SAVE_EVERY,
) -> list[EvaluationRecord]:
    """Прогоняет RAG и LLM-as-judge по набору тестовых запросов.

    Функция сохраняет промежуточный JSONL-файл, если передан ``output_path``.
    Ошибки отдельных запросов не прерывают весь прогон: они фиксируются в
    ``EvaluationRecord.error``.
    """

    if save_every <= 0:
        raise ValueError(f"save_every должен быть положительным числом, получено: {save_every}.")
    active_judge = judge_client or JudgeClient(settings.judge)
    records: list[EvaluationRecord] = []
    for index, evaluation_query in enumerate(queries, start=1):
        try:
            rag_response = run_rag_pipeline(
                evaluation_query.query,
                collection=collection,
                embedding_model=embedding_model,
                bm25_index=bm25_index,
                settings=settings,
                llm_client=llm_client,
                top_k=top_k,
            )
            context = build_rag_context(rag_response.retrieved_chunks)
            judge_result = active_judge.evaluate(
                query=evaluation_query.query,
                context=context,
                answer=rag_response.answer,
                gold=evaluation_query.gold,
                kind=evaluation_query.kind,
            )
            records.append(build_evaluation_record(evaluation_query, rag_response, judge_result=judge_result))
        except Exception as exc:  # noqa: BLE001 - оценочный прогон должен переживать ошибку одного запроса.
            records.append(build_error_record(evaluation_query, str(exc)))
        if output_path and (index % save_every == 0 or index == len(queries)):
            save_evaluation_records_jsonl(records, output_path)
    return records


def build_evaluation_record(
    evaluation_query: EvaluationQuery,
    rag_response: RagResponse,
    *,
    judge_result: JudgeResult | None,
    error: str | None = None,
) -> EvaluationRecord:
    """Собирает полную запись оценки из query, RAG-ответа и judge-result."""

    return EvaluationRecord(
        query=evaluation_query.query,
        kind=evaluation_query.kind,
        relevant_docs=list(evaluation_query.relevant_docs),
        gold=evaluation_query.gold,
        retrieved_chunks=[serialize_retrieved_chunk(chunk) for chunk in rag_response.retrieved_chunks],
        retrieved_doc_ids=[chunk.doc_id for chunk in rag_response.retrieved_chunks],
        retrieval_scores=[serialize_retrieval_score(chunk) for chunk in rag_response.retrieved_chunks],
        answer=rag_response.answer,
        judge_result=judge_result,
        error=error,
        diagnostics=dict(rag_response.diagnostics),
    )


def build_error_record(evaluation_query: EvaluationQuery, error: str) -> EvaluationRecord:
    """Создаёт запись оценки для запроса, на котором pipeline завершился ошибкой."""

    return EvaluationRecord(
        query=evaluation_query.query,
        kind=evaluation_query.kind,
        relevant_docs=list(evaluation_query.relevant_docs),
        gold=evaluation_query.gold,
        retrieved_chunks=[],
        retrieved_doc_ids=[],
        retrieval_scores=[],
        answer="",
        judge_result=None,
        error=error,
        diagnostics={},
    )


def save_evaluation_records_jsonl(records: Sequence[EvaluationRecord], output_path: str | Path) -> None:
    """Сохраняет оценочные записи в JSONL для восстановления после сбоя."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")


def load_evaluation_records_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Загружает ранее сохранённые JSONL-записи как словари."""

    source_path = Path(path)
    if not source_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in source_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def compute_generation_metrics(records: Sequence[EvaluationRecord]) -> GenerationMetricsSummary:
    """Считает generation-метрики по результатам LLM-as-judge."""

    evaluated = [record for record in records if record.judge_result is not None and record.judge_result.is_valid]
    invalid_judge_results = len([record for record in records if record.judge_result is not None and not record.judge_result.is_valid])
    pipeline_errors = len([record for record in records if record.error])
    gap_records = [record for record in records if record.kind == "gap" and record.judge_result is not None and record.judge_result.is_valid]
    gap_hallucinations = [record for record in gap_records if record.judge_result and record.judge_result.correctness == 0]
    return GenerationMetricsSummary(
        queries=len(records),
        evaluated=len(evaluated),
        faithfulness=_round_mean(record.judge_result.faithfulness for record in evaluated if record.judge_result),
        answer_relevance=_round_mean(record.judge_result.answer_relevance for record in evaluated if record.judge_result),
        correctness=_round_mean(record.judge_result.correctness for record in evaluated if record.judge_result),
        invalid_judge_results=invalid_judge_results,
        pipeline_errors=pipeline_errors,
        gap_hallucination_rate=round(len(gap_hallucinations) / len(gap_records), 6) if gap_records else 0.0,
    )


def compute_generation_metrics_by_kind(records: Sequence[EvaluationRecord]) -> dict[str, GenerationMetricsSummary]:
    """Считает generation-метрики отдельно по типам запросов."""

    grouped: dict[str, list[EvaluationRecord]] = {}
    for record in records:
        grouped.setdefault(record.kind, []).append(record)
    return {kind: compute_generation_metrics(kind_records) for kind, kind_records in sorted(grouped.items())}


def build_quality_profile(
    records: Sequence[EvaluationRecord],
    retrieval_results: Sequence[RetrievalResult],
    *,
    k: int = 5,
    recall_target: float = DEFAULT_RECALL_AT_5_TARGET,
    correctness_target: float = DEFAULT_CORRECTNESS_TARGET,
) -> QualityProfile:
    """Собирает сводный профиль retrieval- и generation-качества."""

    retrieval_summary = compute_retrieval_metrics(retrieval_results, k=k)
    generation_summary = compute_generation_metrics(records)
    targets = {
        f"recall@{k}_target": recall_target,
        f"recall@{k}": retrieval_summary.recall_at_k,
        f"recall@{k}_passed": retrieval_summary.recall_at_k >= recall_target,
        "correctness_target": correctness_target,
        "correctness": generation_summary.correctness,
        "correctness_passed": generation_summary.correctness >= correctness_target,
        "gap_hallucination_rate": generation_summary.gap_hallucination_rate,
        "gap_policy_passed": generation_summary.gap_hallucination_rate == 0.0,
    }
    return QualityProfile(
        retrieval=retrieval_summary,
        retrieval_by_kind=compute_retrieval_metrics_by_kind(retrieval_results, k=k),
        generation=generation_summary,
        generation_by_kind=compute_generation_metrics_by_kind(records),
        targets=targets,
    )


def run_golden_context_comparison(
    records: Sequence[EvaluationRecord],
    queries: Sequence[EvaluationQuery],
    documents: Sequence[KnowledgeDocument],
    *,
    llm_client: OpenAIChatClient,
    judge_client: JudgeClient,
    generation_settings: GenerationSettings,
) -> list[GoldenContextRecord]:
    """Сравнивает обычный RAG с генерацией на golden context.

    Для ``gap``-запросов golden context остаётся пустым, чтобы проверить отказ.
    Для остальных запросов генератор получает полные размеченные разделы из
    ``relevant_docs``.
    """

    records_by_query = {record.query: record for record in records}
    documents_by_id = {document.doc_id: document for document in documents}
    comparisons: list[GoldenContextRecord] = []
    for evaluation_query in queries:
        rag_record = records_by_query.get(evaluation_query.query)
        try:
            golden_chunks = build_golden_context_chunks(evaluation_query, documents_by_id)
            golden_answer, _, _ = generate_answer(evaluation_query.query, golden_chunks, llm_client, generation_settings)
            context = build_rag_context(golden_chunks)
            golden_judge = judge_client.evaluate(
                query=evaluation_query.query,
                context=context,
                answer=golden_answer,
                gold=evaluation_query.gold,
                kind=evaluation_query.kind,
            )
            rag_correctness = rag_record.judge_result.correctness if rag_record and rag_record.judge_result else None
            golden_correctness = golden_judge.correctness if golden_judge.is_valid else None
            delta = None if rag_correctness is None or golden_correctness is None else golden_correctness - rag_correctness
            comparisons.append(
                GoldenContextRecord(
                    query=evaluation_query.query,
                    kind=evaluation_query.kind,
                    relevant_docs=list(evaluation_query.relevant_docs),
                    gold=evaluation_query.gold,
                    rag_answer=rag_record.answer if rag_record else "",
                    golden_answer=golden_answer,
                    rag_correctness=rag_correctness,
                    golden_correctness=golden_correctness,
                    delta=delta,
                    error=golden_judge.parse_error,
                )
            )
        except Exception as exc:  # noqa: BLE001 - сравнение должно сохранить ошибку конкретного запроса.
            comparisons.append(
                GoldenContextRecord(
                    query=evaluation_query.query,
                    kind=evaluation_query.kind,
                    relevant_docs=list(evaluation_query.relevant_docs),
                    gold=evaluation_query.gold,
                    rag_answer=rag_record.answer if rag_record else "",
                    golden_answer="",
                    rag_correctness=rag_record.judge_result.correctness if rag_record and rag_record.judge_result else None,
                    golden_correctness=None,
                    delta=None,
                    error=str(exc),
                )
            )
    return comparisons


def build_golden_context_chunks(
    evaluation_query: EvaluationQuery,
    documents_by_id: dict[int, KnowledgeDocument],
) -> list[RetrievedChunk]:
    """Строит pseudo-retrieval chunks из эталонных документов запроса."""

    if evaluation_query.kind == "gap" or not evaluation_query.relevant_docs:
        return []
    chunks: list[RetrievedChunk] = []
    for rank, doc_id in enumerate(evaluation_query.relevant_docs, start=1):
        document = documents_by_id.get(doc_id)
        if document is None:
            continue
        chunks.append(
            RetrievedChunk(
                chunk_id=f"golden_doc_{doc_id}",
                doc_id=doc_id,
                text=document.raw,
                metadata={
                    "doc_id": document.doc_id,
                    "title": document.title,
                    "category": document.category,
                    "format": document.format,
                    "chunk_id": f"golden_doc_{doc_id}",
                    "chunk_index": 0,
                    "chunking_strategy": "golden_context",
                },
                distance=None,
                score=1.0,
                rank=rank,
                retrieval_method="golden_context",
            )
        )
    return chunks


def build_error_analysis(records: Sequence[EvaluationRecord], *, limit: int = DEFAULT_WORST_CASE_LIMIT) -> list[dict[str, Any]]:
    """Выбирает худшие запросы и добавляет диагноз причины ошибки."""

    if limit <= 0:
        raise ValueError(f"limit должен быть положительным числом, получено: {limit}.")
    ranked_records = sorted(records, key=_error_severity, reverse=True)
    rows: list[dict[str, Any]] = []
    for record in ranked_records[:limit]:
        rows.append(
            {
                "query": record.query,
                "kind": record.kind,
                "gold": record.gold,
                "relevant_docs": list(record.relevant_docs),
                "retrieved_doc_ids": list(record.retrieved_doc_ids),
                "retrieved_context": record.retrieved_chunks,
                "answer": record.answer,
                "judge_comment": record.judge_result.comment if record.judge_result else "",
                "diagnosis": diagnose_evaluation_error(record),
                "error": record.error,
            }
        )
    return rows


def diagnose_evaluation_error(record: EvaluationRecord) -> str:
    """Классифицирует вероятную причину ошибки оценочного запроса."""

    if record.error:
        return "Ошибка выполнения RAG/LLM-вызова."
    if record.judge_result and not record.judge_result.is_valid:
        return "Судья вернул невалидную оценку."
    relevant = set(record.relevant_docs)
    retrieved = set(record.retrieved_doc_ids)
    if record.kind != "gap" and relevant and not relevant.intersection(retrieved):
        return "Неверный retrieval: релевантный документ не найден в top-K."
    if record.kind != "gap" and record.judge_result and record.judge_result.correctness == 0:
        return "Контекст найден, но генератор или судья дали некорректный результат."
    if record.kind == "gap" and record.judge_result and record.judge_result.correctness == 0:
        return "Gap-запрос: ответ содержит неподтверждённые факты или отказ не распознан."
    return "Критической ошибки не выявлено; пример полезен для ручной проверки."


def build_final_verdict(
    quality_profile: QualityProfile,
    *,
    min_recommendations: int = 3,
) -> dict[str, Any]:
    """Формирует инженерный вердикт о готовности системы к пилоту."""

    targets = quality_profile.targets
    ready = bool(targets.get("recall@5_passed") and targets.get("correctness_passed") and targets.get("gap_policy_passed"))
    recommendations = [
        "Улучшить retrieval для достижения recall@5 >= 0.85: подобрать top-K/fetch-K, reranker и пороги RRF.",
        "Использовать сохранённый полный прогон 66 запросов как baseline и повторять его после каждого изменения retrieval/generation.",
        "Разобрать худшие запросы по grouped error analysis и обновить очистку/чанкинг для повторяющихся причин ошибок.",
        "Синхронизировать notebook-отчёт с модульным кодом этапа 6 и добавить golden-context сравнение.",
    ]
    return {
        "ready_for_pilot": ready,
        "reason": "Целевые ориентиры достигнуты." if ready else "Пилот преждевременен до подтверждения целевых retrieval/generation-метрик.",
        "targets": dict(targets),
        "recommendations": recommendations[: max(min_recommendations, 0)],
    }


def serialize_retrieved_chunk(chunk: RetrievedChunk, *, text_preview_chars: int = 500) -> dict[str, Any]:
    """Сериализует найденный чанк для таблицы результатов."""

    return {
        "rank": chunk.rank,
        "doc_id": chunk.doc_id,
        "chunk_id": chunk.chunk_id,
        "title": chunk.metadata.get("title", ""),
        "category": chunk.metadata.get("category", ""),
        "method": chunk.retrieval_method,
        "score": chunk.score,
        "distance": chunk.distance,
        "text_preview": _trim_text(chunk.text, text_preview_chars),
    }


def serialize_retrieval_score(chunk: RetrievedChunk) -> dict[str, Any]:
    """Сериализует score/distance одного retrieval-результата."""

    return {
        "rank": chunk.rank,
        "doc_id": chunk.doc_id,
        "chunk_id": chunk.chunk_id,
        "score": chunk.score,
        "distance": chunk.distance,
        "source_scores": dict(chunk.source_scores or {}),
        "source_ranks": dict(chunk.source_ranks or {}),
    }


def build_retrieval_results_from_records(records: Sequence[EvaluationRecord]) -> list[RetrievalResult]:
    """Восстанавливает ``RetrievalResult`` из оценочных записей для метрик."""

    results: list[RetrievalResult] = []
    for record in records:
        chunks = [
            RetrievedChunk(
                chunk_id=str(chunk.get("chunk_id", f"record_{row_index}")),
                doc_id=int(chunk.get("doc_id", -1)),
                text=str(chunk.get("text_preview", "")),
                metadata={
                    "title": chunk.get("title", ""),
                    "category": chunk.get("category", ""),
                    "chunk_id": chunk.get("chunk_id", f"record_{row_index}"),
                },
                distance=chunk.get("distance"),
                score=chunk.get("score"),
                rank=int(chunk.get("rank", row_index)),
                retrieval_method=str(chunk.get("method", "record")),
            )
            for row_index, chunk in enumerate(record.retrieved_chunks, start=1)
        ]
        results.append(
            RetrievalResult(query=record.query, kind=record.kind, relevant_docs=list(record.relevant_docs), retrieved_chunks=chunks)
        )
    return results


def validate_judge_settings(settings: JudgeSettings) -> None:
    """Проверяет настройки LLM-судьи перед сетевым вызовом."""

    _require_non_empty_string(settings.base_url, "JUDGE_BASE_URL")
    _require_non_empty_string(settings.api_key, "JUDGE_API_KEY")
    _require_non_empty_string(settings.model_name, "JUDGE_MODEL_NAME")
    if float(settings.temperature) != 0.0:
        raise ValueError(f"JUDGE_TEMPERATURE должен быть равен 0 для воспроизводимой оценки, получено: {settings.temperature}.")
    if int(settings.timeout_seconds) <= 0:
        raise ValueError(f"JUDGE_TIMEOUT_SECONDS должен быть положительным, получено: {settings.timeout_seconds}.")
    if int(settings.max_tokens) <= 0:
        raise ValueError(f"JUDGE_MAX_TOKENS должен быть положительным, получено: {settings.max_tokens}.")


def _judge_settings_to_llm_settings(settings: JudgeSettings) -> Any:
    """Преобразует ``JudgeSettings`` в совместимый с ``OpenAIChatClient`` объект."""

    from src.config import LlmSettings

    return LlmSettings(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model_name=settings.model_name,
        temperature=settings.temperature,
        timeout_seconds=settings.timeout_seconds,
        max_tokens=settings.max_tokens,
    )


def _parse_binary_score(value: Any, field_name: str) -> int:
    """Парсит бинарную оценку judge-метрики."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in {0, 1}:
        return value
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return int(value.strip())
    raise ValueError(f"Поле {field_name!r} должно быть бинарным 0/1, получено: {value!r}.")


def _round_mean(values: Sequence[int] | Any) -> float:
    """Считает среднее и округляет до 6 знаков."""

    value_list = [float(value) for value in values]
    if not value_list:
        return 0.0
    return round(sum(value_list) / len(value_list), 6)


def _error_severity(record: EvaluationRecord) -> tuple[int, int, int, float]:
    """Возвращает сортировочный ключ худших оценочных записей."""

    correctness_error = 1 if record.judge_result and record.judge_result.correctness == 0 else 0
    invalid_judge = 1 if record.judge_result and not record.judge_result.is_valid else 0
    pipeline_error = 1 if record.error else 0
    retrieval_miss = 0.0
    if record.kind != "gap" and record.relevant_docs:
        retrieval_miss = 1.0 - recall_at_k(record.retrieved_doc_ids, record.relevant_docs, max(1, len(record.retrieved_doc_ids)))
    return (pipeline_error, invalid_judge, correctness_error, retrieval_miss)


def _trim_text(text: str, limit: int) -> str:
    """Обрезает текст для диагностического вывода."""

    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "…"


def _require_non_empty_string(value: str, name: str) -> str:
    """Проверяет непустую строковую настройку."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} должен быть непустой строкой.")
    return value.strip()


__all__ = [
    "DEFAULT_CORRECTNESS_TARGET",
    "DEFAULT_INTERMEDIATE_SAVE_EVERY",
    "DEFAULT_JUDGE_RETRY_COUNT",
    "DEFAULT_RECALL_AT_5_TARGET",
    "DEFAULT_WORST_CASE_LIMIT",
    "EvaluationRecord",
    "GenerationMetricsSummary",
    "GoldenContextRecord",
    "JudgeClient",
    "JudgeResult",
    "QualityProfile",
    "build_error_analysis",
    "build_evaluation_record",
    "build_failed_judge_result",
    "build_final_verdict",
    "build_golden_context_chunks",
    "build_judge_prompt",
    "build_quality_profile",
    "build_retrieval_results_from_records",
    "compute_generation_metrics",
    "compute_generation_metrics_by_kind",
    "diagnose_evaluation_error",
    "extract_json_object",
    "load_evaluation_records_jsonl",
    "parse_judge_response",
    "run_evaluation_queries",
    "run_golden_context_comparison",
    "safe_parse_judge_response",
    "save_evaluation_records_jsonl",
    "serialize_retrieval_score",
    "serialize_retrieved_chunk",
    "validate_judge_settings",
]