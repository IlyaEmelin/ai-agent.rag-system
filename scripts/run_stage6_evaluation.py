"""CLI-запуск этапа 6: оценка RAG, golden-context и финальный вердикт.

Скрипт собирает runtime-объекты проекта тем же способом, который требуется в
notebook: настройки, датасет, очищенные документы, recursive chunks, embedding
wrapper, Chroma collection и BM25Index. Затем он запускает LLM-as-judge оценку,
строит профиль качества, golden-context сравнение, диагностику ошибок и итоговый
инженерный вердикт.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.chunking import build_chunking_strategies_from_settings, build_document_chunks, build_langchain_documents
from src.cleaning import clean_corpus
from src.config import PROJECT_ROOT, Settings, load_settings
from src.data_loading import EvaluationQuery, load_dataset
from src.embeddings import SentenceTransformerEmbeddingModel
from src.generation import OpenAIChatClient
from src.judging import (
    JudgeClient,
    build_error_analysis,
    build_final_verdict,
    build_quality_profile,
    build_retrieval_results_from_records,
    run_evaluation_queries,
    run_golden_context_comparison,
)
from src.retrieval import build_bm25_index
from src.vector_store import build_main_collection_name, build_persistent_chroma_client, recreate_main_chroma_collection


DEFAULT_STAGE6_OUTPUT_DIR = PROJECT_ROOT / "temp" / "stage6"
DEFAULT_STAGE6_STRATEGY = "recursive"


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки для stage 6 runner."""

    parser = argparse.ArgumentParser(description="Запуск оценки качества RAG для этапа 6.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_STAGE6_OUTPUT_DIR,
        help="Директория для JSON/JSONL артефактов этапа 6.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничить число запросов для smoke/debug запуска. По умолчанию обрабатываются все запросы.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Переопределить top-K контекста. По умолчанию берётся settings.retrieval.top_k.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Как часто сохранять промежуточный JSONL оценочного прогона.",
    )
    parser.add_argument(
        "--recreate-chroma",
        action="store_true",
        help="Пересоздать основную Chroma collection из текущих recursive chunks и embeddings.",
    )
    parser.add_argument(
        "--skip-golden",
        action="store_true",
        help="Не запускать golden-context comparison. Полезно для быстрого smoke-прогона.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Собрать датасет, очистку, chunks и BM25 без embeddings, Chroma и LLM-вызовов.",
    )
    return parser.parse_args()


def main() -> None:
    """Запускает полный или ограниченный stage 6 evaluation workflow."""

    args = parse_args()
    settings = load_settings()
    dataset = load_dataset(settings.data.dataset_path)
    queries = select_queries(dataset.queries, args.limit)

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_results = clean_corpus(dataset.docs)
    langchain_documents = build_langchain_documents(cleaned_results)
    recursive_strategy = build_chunking_strategies_from_settings(settings.chunking)[0]
    if recursive_strategy.name != DEFAULT_STAGE6_STRATEGY:
        raise RuntimeError(f"Ожидалась recursive-стратегия чанкинга, получено: {recursive_strategy.name!r}.")
    chunks = build_document_chunks(langchain_documents, recursive_strategy)
    bm25_index = build_bm25_index(chunks)

    if args.dry_run:
        summary = build_run_summary(
            settings=settings,
            query_count=len(queries),
            chunk_count=len(chunks),
            collection_name=build_main_collection_name(settings.chroma.collection_prefix, DEFAULT_STAGE6_STRATEGY),
            collection_count=None,
            bm25_document_count=len(bm25_index.chunks),
            eval_path=output_dir / "stage6_eval_results.jsonl",
            artifact_paths={"summary_path": output_dir / "stage6_summary.json"},
            skip_golden=bool(args.skip_golden),
            dry_run=True,
        )
        write_json(output_dir / "stage6_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    embedding_model = SentenceTransformerEmbeddingModel(settings.embedding)
    collection = load_or_recreate_main_collection(
        settings,
        embedding_model,
        chunks,
        recreate=bool(args.recreate_chroma),
    )
    llm_client = OpenAIChatClient(settings.llm)
    judge_client = JudgeClient(settings.judge)

    eval_path = output_dir / "stage6_eval_results.jsonl"
    records = run_evaluation_queries(
        queries,
        collection=collection,
        embedding_model=embedding_model,
        bm25_index=bm25_index,
        settings=settings,
        llm_client=llm_client,
        judge_client=judge_client,
        top_k=args.top_k,
        output_path=eval_path,
        save_every=args.save_every,
    )

    retrieval_results = build_retrieval_results_from_records(records)
    quality_profile = build_quality_profile(records, retrieval_results)
    verdict = build_final_verdict(quality_profile)
    error_analysis = build_error_analysis(records)

    golden_records = []
    if not args.skip_golden:
        golden_records = run_golden_context_comparison(
            records,
            queries,
            dataset.docs,
            llm_client=llm_client,
            judge_client=judge_client,
            generation_settings=settings.generation,
        )

    artifacts = {
        "quality_profile_path": output_dir / "stage6_quality_profile.json",
        "error_analysis_path": output_dir / "stage6_error_analysis.json",
        "final_verdict_path": output_dir / "stage6_final_verdict.json",
        "summary_path": output_dir / "stage6_summary.json",
    }
    if golden_records:
        artifacts["golden_context_path"] = output_dir / "stage6_golden_context.json"

    write_json(artifacts["quality_profile_path"], quality_profile.as_dict())
    write_json(artifacts["error_analysis_path"], error_analysis)
    write_json(artifacts["final_verdict_path"], verdict)
    if golden_records:
        write_json(artifacts["golden_context_path"], [record.as_dict() for record in golden_records])

    summary = build_run_summary(
        settings=settings,
        query_count=len(queries),
        chunk_count=len(chunks),
        collection_name=collection.name,
        collection_count=collection.count(),
        bm25_document_count=len(bm25_index.chunks),
        eval_path=eval_path,
        artifact_paths=artifacts,
        skip_golden=bool(args.skip_golden),
        dry_run=False,
    )
    write_json(artifacts["summary_path"], summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def select_queries(queries: Sequence[EvaluationQuery], limit: int | None) -> list[EvaluationQuery]:
    """Возвращает все запросы или первые ``limit`` запросов для smoke-запуска."""

    if limit is None:
        return list(queries)
    if limit <= 0:
        raise ValueError(f"--limit должен быть положительным числом, получено: {limit}.")
    return list(queries)[:limit]


def resolve_output_dir(path: Path) -> Path:
    """Возвращает абсолютный путь директории артефактов."""

    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return PROJECT_ROOT / expanded


def load_or_recreate_main_collection(
    settings: Settings,
    embedding_model: SentenceTransformerEmbeddingModel,
    chunks: Sequence[Any],
    *,
    recreate: bool,
) -> Any:
    """Подключает существующую Chroma collection или пересоздаёт её."""

    collection_name = build_main_collection_name(settings.chroma.collection_prefix, DEFAULT_STAGE6_STRATEGY)
    if recreate:
        embeddings = embedding_model.embed_documents(chunks)
        return recreate_main_chroma_collection(settings.chroma, DEFAULT_STAGE6_STRATEGY, chunks, embeddings)

    client = build_persistent_chroma_client(settings.chroma.persist_dir)
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    if collection.count() != len(chunks):
        raise RuntimeError(
            "Основная Chroma collection не синхронизирована с текущими chunks: "
            f"collection_count={collection.count()}, chunk_count={len(chunks)}. "
            "Перезапустите скрипт с --recreate-chroma."
        )
    return collection


def build_run_summary(
    *,
    settings: Settings,
    query_count: int,
    chunk_count: int,
    collection_name: str,
    collection_count: int | None,
    bm25_document_count: int,
    eval_path: Path,
    artifact_paths: dict[str, Path],
    skip_golden: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Строит краткую сводку запуска для консоли и JSON-файла."""

    return {
        "query_count": query_count,
        "chunk_count": chunk_count,
        "collection_name": collection_name,
        "collection_count": collection_count,
        "bm25_document_count": bm25_document_count,
        "embedding_model": settings.embedding.model_name,
        "llm_model": settings.llm.model_name,
        "judge_model": settings.judge.model_name,
        "top_k": settings.retrieval.top_k,
        "golden_context_skipped": skip_golden,
        "dry_run": dry_run,
        "eval_results_path": str(eval_path),
        "artifacts": {name: str(path) for name, path in artifact_paths.items()},
    }


def write_json(path: Path, payload: Any) -> None:
    """Сохраняет JSON с UTF-8 и стабильным форматированием."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    """Преобразует dataclass/path/tuple структуры в JSON-совместимый вид."""

    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()