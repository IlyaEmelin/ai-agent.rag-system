"""Утилиты загрузки датасета базы знаний «Фильмоскоп»."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_DOCUMENT_COUNT = 181
EXPECTED_QUERY_COUNT = 66
REQUIRED_DOCUMENT_FIELDS = frozenset({"doc_id", "title", "category", "format", "raw"})
REQUIRED_QUERY_FIELDS = frozenset({"query", "kind", "relevant_docs", "gold"})


@dataclass(frozen=True)
class KnowledgeDocument:
    """Документ базы знаний «Фильмоскоп».

    Атрибуты:
        doc_id: Целочисленный идентификатор документа в датасете.
        title: Заголовок документа.
        category: Категория документа.
        format: Исходный формат поля ``raw``: ``text``, ``markdown`` или ``html``.
        raw: Исходный текст документа без очистки.
    """

    doc_id: int
    title: str
    category: str
    format: str
    raw: str


@dataclass(frozen=True)
class EvaluationQuery:
    """Тестовый запрос для оценки RAG-пайплайна.

    Атрибуты:
        query: Текст пользовательского вопроса.
        kind: Тип запроса: ``exact``, ``semantic``, ``multi`` или ``gap``.
        relevant_docs: Список релевантных ``doc_id`` для retrieval-оценки.
        gold: Эталонный ответ или краткое описание ожидаемого поведения.
    """

    query: str
    kind: str
    relevant_docs: list[int]
    gold: str


@dataclass(frozen=True)
class FilmoskopDataset:
    """Загруженный датасет базы знаний и тестовых запросов.

    Атрибуты:
        docs: Документы базы знаний.
        queries: Тестовые запросы с эталонными данными.
        source_path: Абсолютный путь к JSON-файлу, из которого загружен датасет.
    """

    docs: list[KnowledgeDocument]
    queries: list[EvaluationQuery]
    source_path: Path


def load_dataset(path: str | Path) -> FilmoskopDataset:
    """Загружает и валидирует датасет «Фильмоскоп».

    Аргументы:
        path: Путь к JSON-файлу датасета. Обычно берётся из
            ``settings.data.dataset_path``.

    Возвращает:
        Валидированный датасет с разделёнными ``docs`` и ``queries``.

    Исключения:
        FileNotFoundError: Если файл датасета отсутствует.
        ValueError: Если JSON некорректен или структура датасета не соответствует
            ожидаемой схеме проекта.
    """

    dataset_path = Path(path).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Файл датасета не найден: {dataset_path}. Проверьте значение DATASET_PATH в .env."
        )
    if not dataset_path.is_file():
        raise ValueError(f"Путь датасета должен указывать на файл, получено: {dataset_path}")

    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Не удалось разобрать JSON датасета {dataset_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Корневой элемент датасета должен быть JSON-объектом с ключами 'docs' и 'queries'.")

    raw_docs = _require_list(payload, "docs")
    raw_queries = _require_list(payload, "queries")
    docs = [_parse_document(item, index) for index, item in enumerate(raw_docs)]
    queries = [_parse_query(item, index) for index, item in enumerate(raw_queries)]
    _validate_dataset_counts(docs, queries)
    _validate_unique_doc_ids(docs)
    _validate_query_references(queries, docs)

    return FilmoskopDataset(docs=docs, queries=queries, source_path=dataset_path.resolve())


def load_filmoskop_dataset(path: str | Path) -> FilmoskopDataset:
    """Загружает датасет «Фильмоскоп» через основную функцию ``load_dataset``.

    Аргументы:
        path: Путь к JSON-файлу датасета.

    Возвращает:
        Валидированный датасет с документами и тестовыми запросами.
    """

    return load_dataset(path)


def _require_list(payload: dict[str, Any], key: str) -> list[Any]:
    """Возвращает обязательный список из корневого JSON-объекта.

    Аргументы:
        payload: Корневой JSON-объект датасета.
        key: Имя обязательного поля.

    Возвращает:
        Значение поля как список.

    Исключения:
        ValueError: Если поле отсутствует или не является списком.
    """

    if key not in payload:
        raise ValueError(f"В датасете отсутствует обязательный раздел {key!r}.")
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"Раздел {key!r} должен быть списком, получено: {type(value).__name__}.")
    return value


def _parse_document(item: Any, index: int) -> KnowledgeDocument:
    """Преобразует JSON-объект документа в ``KnowledgeDocument``.

    Аргументы:
        item: JSON-значение одного документа.
        index: Позиция документа в массиве ``docs`` для понятных ошибок.

    Возвращает:
        Типизированный документ базы знаний.
    """

    if not isinstance(item, dict):
        raise ValueError(f"Документ docs[{index}] должен быть объектом, получено: {type(item).__name__}.")
    _validate_required_fields(item, REQUIRED_DOCUMENT_FIELDS, f"docs[{index}]")

    doc_id = item["doc_id"]
    title = item["title"]
    category = item["category"]
    document_format = item["format"]
    raw = item["raw"]

    if not isinstance(doc_id, int):
        raise ValueError(f"Поле docs[{index}].doc_id должно быть целым числом, получено: {doc_id!r}.")
    for field_name, value in {
        "title": title,
        "category": category,
        "format": document_format,
        "raw": raw,
    }.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Поле docs[{index}].{field_name} должно быть непустой строкой.")

    return KnowledgeDocument(doc_id=doc_id, title=title, category=category, format=document_format, raw=raw)


def _parse_query(item: Any, index: int) -> EvaluationQuery:
    """Преобразует JSON-объект запроса в ``EvaluationQuery``.

    Аргументы:
        item: JSON-значение одного запроса.
        index: Позиция запроса в массиве ``queries`` для понятных ошибок.

    Возвращает:
        Типизированный тестовый запрос.
    """

    if not isinstance(item, dict):
        raise ValueError(f"Запрос queries[{index}] должен быть объектом, получено: {type(item).__name__}.")
    _validate_required_fields(item, REQUIRED_QUERY_FIELDS, f"queries[{index}]")

    query = item["query"]
    kind = item["kind"]
    relevant_docs = item["relevant_docs"]
    gold = item["gold"]

    for field_name, value in {"query": query, "kind": kind, "gold": gold}.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Поле queries[{index}].{field_name} должно быть непустой строкой.")
    if not isinstance(relevant_docs, list):
        raise ValueError(f"Поле queries[{index}].relevant_docs должно быть списком.")
    if not all(isinstance(doc_id, int) for doc_id in relevant_docs):
        raise ValueError(f"Все значения queries[{index}].relevant_docs должны быть целыми doc_id.")

    return EvaluationQuery(query=query, kind=kind, relevant_docs=list(relevant_docs), gold=gold)


def _validate_required_fields(item: dict[str, Any], required_fields: frozenset[str], label: str) -> None:
    """Проверяет наличие обязательных полей в JSON-объекте.

    Аргументы:
        item: Проверяемый JSON-объект.
        required_fields: Множество обязательных полей.
        label: Человекочитаемая метка объекта для сообщения об ошибке.

    Исключения:
        ValueError: Если отсутствует хотя бы одно обязательное поле.
    """

    missing_fields = sorted(required_fields.difference(item))
    if missing_fields:
        raise ValueError(f"В объекте {label} отсутствуют обязательные поля: {', '.join(missing_fields)}.")


def _validate_dataset_counts(docs: list[KnowledgeDocument], queries: list[EvaluationQuery]) -> None:
    """Проверяет ожидаемое количество документов и запросов.

    Аргументы:
        docs: Список документов базы знаний.
        queries: Список тестовых запросов.

    Исключения:
        ValueError: Если размеры датасета отличаются от ожидаемых значений проекта.
    """

    if len(docs) != EXPECTED_DOCUMENT_COUNT:
        raise ValueError(f"Ожидалось документов: {EXPECTED_DOCUMENT_COUNT}, получено: {len(docs)}.")
    if len(queries) != EXPECTED_QUERY_COUNT:
        raise ValueError(f"Ожидалось запросов: {EXPECTED_QUERY_COUNT}, получено: {len(queries)}.")


def _validate_unique_doc_ids(docs: list[KnowledgeDocument]) -> None:
    """Проверяет уникальность ``doc_id`` в корпусе.

    Аргументы:
        docs: Список документов базы знаний.

    Исключения:
        ValueError: Если один и тот же ``doc_id`` встречается несколько раз.
    """

    seen_doc_ids: set[int] = set()
    duplicated_doc_ids: set[int] = set()
    for document in docs:
        if document.doc_id in seen_doc_ids:
            duplicated_doc_ids.add(document.doc_id)
        seen_doc_ids.add(document.doc_id)
    if duplicated_doc_ids:
        formatted_doc_ids = ", ".join(str(doc_id) for doc_id in sorted(duplicated_doc_ids))
        raise ValueError(f"В датасете найдены повторяющиеся doc_id: {formatted_doc_ids}.")


def _validate_query_references(queries: list[EvaluationQuery], docs: list[KnowledgeDocument]) -> None:
    """Проверяет, что ссылки ``relevant_docs`` указывают на существующие документы.

    Аргументы:
        queries: Список тестовых запросов.
        docs: Список документов базы знаний.

    Исключения:
        ValueError: Если запрос ссылается на отсутствующий ``doc_id``.
    """

    existing_doc_ids = {document.doc_id for document in docs}
    errors: list[str] = []
    for index, query in enumerate(queries):
        missing_doc_ids = sorted(set(query.relevant_docs).difference(existing_doc_ids))
        if missing_doc_ids:
            formatted_doc_ids = ", ".join(str(doc_id) for doc_id in missing_doc_ids)
            errors.append(f"queries[{index}]: {formatted_doc_ids}")
    if errors:
        raise ValueError("Найдены ссылки на отсутствующие документы в relevant_docs: " + "; ".join(errors))


__all__ = [
    "EXPECTED_DOCUMENT_COUNT",
    "EXPECTED_QUERY_COUNT",
    "EvaluationQuery",
    "FilmoskopDataset",
    "KnowledgeDocument",
    "REQUIRED_DOCUMENT_FIELDS",
    "REQUIRED_QUERY_FIELDS",
    "load_dataset",
    "load_filmoskop_dataset",
]
