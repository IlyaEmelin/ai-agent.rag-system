"""Утилиты диагностики и подготовки отчётов для RAG-экспериментов."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable, Sequence

from src.cleaning import (
    get_text_quality_flags,
    normalize_text_for_duplicate_detection,
    text_similarity,
)
from src.data_loading import KnowledgeDocument


DEFAULT_SHORT_QUANTILE = 0.05
DEFAULT_LONG_QUANTILE = 0.95
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.94


@dataclass(frozen=True)
class AuditTable:
    """Лёгкая табличная структура для аудита без обязательной зависимости от pandas.

    Атрибуты:
        rows: Строки таблицы как словари ``имя колонки → значение``.
        columns: Порядок колонок для вывода.
    """

    rows: tuple[dict[str, Any], ...]
    columns: tuple[str, ...]

    def __len__(self) -> int:
        """Возвращает количество строк таблицы."""

        return len(self.rows)

    def __iter__(self) -> Iterable[dict[str, Any]]:
        """Позволяет итерироваться по строкам таблицы."""

        return iter(self.rows)

    def __bool__(self) -> bool:
        """Проверяет, есть ли в таблице строки."""

        return bool(self.rows)

    def __getitem__(self, key: str | Sequence[str]) -> list[Any] | "AuditTable":
        """Возвращает колонку или таблицу с выбранными колонками.

        Аргументы:
            key: Имя колонки или последовательность имён колонок.

        Возвращает:
            Список значений колонки или новую таблицу.
        """

        if isinstance(key, str):
            return [row.get(key) for row in self.rows]
        return self.select(key)

    def head(self, count: int = 5) -> "AuditTable":
        """Возвращает первые строки таблицы.

        Аргументы:
            count: Максимальное число строк.

        Возвращает:
            Новую таблицу с первыми ``count`` строками.
        """

        return AuditTable(rows=self.rows[:count], columns=self.columns)

    def select(self, columns: Sequence[str]) -> "AuditTable":
        """Возвращает таблицу только с выбранными колонками.

        Аргументы:
            columns: Имена колонок в нужном порядке.

        Возвращает:
            Новую таблицу с указанными колонками.
        """

        selected_columns = tuple(columns)
        selected_rows = tuple({column: row.get(column, "") for column in selected_columns} for row in self.rows)
        return AuditTable(rows=selected_rows, columns=selected_columns)

    def to_string(self, *, index: bool = False, max_rows: int | None = None) -> str:
        """Форматирует таблицу в человекочитаемую строку.

        Аргументы:
            index: Добавлять ли колонку с номером строки.
            max_rows: Максимальное число строк для вывода.

        Возвращает:
            Текстовая таблица с выровненными колонками.
        """

        rows = self.rows if max_rows is None else self.rows[:max_rows]
        columns = (("index",) if index else tuple()) + self.columns
        display_rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            display_row = dict(row)
            if index:
                display_row["index"] = row_index
            display_rows.append(display_row)
        widths = {
            column: max(
                len(str(column)),
                *(len(_stringify(row.get(column, ""))) for row in display_rows),
            )
            for column in columns
        }
        header = "  ".join(str(column).ljust(widths[column]) for column in columns)
        separator = "  ".join("-" * widths[column] for column in columns)
        lines = [header, separator]
        for row in display_rows:
            lines.append("  ".join(_stringify(row.get(column, "")).ljust(widths[column]) for column in columns))
        if not display_rows:
            lines.append("<нет строк>")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Возвращает HTML-представление для Jupyter Notebook."""

        header = "".join(f"<th>{_escape_html(column)}</th>" for column in self.columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{_escape_html(_stringify(row.get(column, '')))}</td>" for column in self.columns) + "</tr>"
            for row in self.rows
        )
        return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"

    def __repr__(self) -> str:
        """Возвращает компактное текстовое представление таблицы."""

        return self.to_string(max_rows=20)


@dataclass(frozen=True)
class DuplicateGroup:
    """Группа полных или почти полных дублей.

    Атрибуты:
        doc_ids: Идентификаторы документов в группе.
        titles: Заголовки документов в группе.
        similarity: Минимальная зафиксированная похожесть внутри группы.
        match_type: Тип группы: ``exact`` или ``near``.
    """

    doc_ids: tuple[int, ...]
    titles: tuple[str, ...]
    similarity: float
    match_type: str


@dataclass(frozen=True)
class CorpusAudit:
    """Результат первичного аудита корпуса.

    Атрибуты:
        total_documents: Количество документов в корпусе.
        category_distribution: Таблица распределения по категориям.
        format_distribution: Таблица распределения по форматам.
        length_statistics: Таблица статистик длины поля ``raw``.
        document_profile: Таблица с профилем каждого документа.
        short_documents: Таблица самых коротких документов.
        long_documents: Таблица самых длинных документов.
        html_documents: Таблица документов с HTML-признаками.
        markdown_documents: Таблица документов с Markdown-признаками.
        mojibake_documents: Таблица документов с признаками mojibake.
        suspicious_documents: Таблица документов с мусорными последовательностями.
        duplicate_groups: Список групп полных или почти полных дублей.
        decisions: Таблица решений по найденным проблемам.
    """

    total_documents: int
    category_distribution: AuditTable
    format_distribution: AuditTable
    length_statistics: AuditTable
    document_profile: AuditTable
    short_documents: AuditTable
    long_documents: AuditTable
    html_documents: AuditTable
    markdown_documents: AuditTable
    mojibake_documents: AuditTable
    suspicious_documents: AuditTable
    duplicate_groups: list[DuplicateGroup]
    decisions: AuditTable


def audit_corpus(
    documents: Sequence[KnowledgeDocument],
    *,
    short_quantile: float = DEFAULT_SHORT_QUANTILE,
    long_quantile: float = DEFAULT_LONG_QUANTILE,
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> CorpusAudit:
    """Выполняет первичный аудит корпуса базы знаний.

    Аргументы:
        documents: Документы, загруженные из датасета.
        short_quantile: Нижний квантиль для отбора коротких документов.
        long_quantile: Верхний квантиль для отбора длинных документов.
        near_duplicate_threshold: Порог похожести для почти полных дублей.

    Возвращает:
        Структуру с таблицами аудита и рекомендованными решениями.
    """

    document_profile = build_document_profile(documents)
    profile_rows = list(document_profile.rows)
    lengths = [int(row["raw_length"]) for row in profile_rows]
    short_threshold = _quantile(lengths, short_quantile)
    long_threshold = _quantile(lengths, long_quantile)
    short_documents = _make_table(
        sorted((row for row in profile_rows if int(row["raw_length"]) <= short_threshold), key=lambda row: int(row["raw_length"])),
        document_profile.columns,
    )
    long_documents = _make_table(
        sorted(
            (row for row in profile_rows if int(row["raw_length"]) >= long_threshold),
            key=lambda row: int(row["raw_length"]),
            reverse=True,
        ),
        document_profile.columns,
    )
    duplicate_groups = find_duplicate_groups(documents, near_duplicate_threshold=near_duplicate_threshold)
    html_documents = _filter_table(document_profile, lambda row: bool(row["has_html"]))
    markdown_documents = _filter_table(document_profile, lambda row: bool(row["has_markdown"]))
    mojibake_documents = _filter_table(document_profile, lambda row: row["mojibake_markers"] != "")
    suspicious_documents = _filter_table(document_profile, lambda row: row["suspicious_sequences"] != "")

    return CorpusAudit(
        total_documents=len(documents),
        category_distribution=build_distribution_table(document_profile, "category"),
        format_distribution=build_distribution_table(document_profile, "format"),
        length_statistics=build_length_statistics(document_profile),
        document_profile=document_profile,
        short_documents=short_documents,
        long_documents=long_documents,
        html_documents=html_documents,
        markdown_documents=markdown_documents,
        mojibake_documents=mojibake_documents,
        suspicious_documents=suspicious_documents,
        duplicate_groups=duplicate_groups,
        decisions=build_problem_decisions(
            html_count=len(html_documents),
            markdown_count=len(markdown_documents),
            mojibake_count=len(mojibake_documents),
            suspicious_count=len(suspicious_documents),
            duplicate_group_count=len(duplicate_groups),
            short_count=len(short_documents),
            long_count=len(long_documents),
        ),
    )


def build_document_profile(documents: Sequence[KnowledgeDocument]) -> AuditTable:
    """Строит таблицу признаков по каждому документу корпуса.

    Аргументы:
        documents: Документы базы знаний.

    Возвращает:
        Таблицу с длиной текста и признаками качества.
    """

    rows: list[dict[str, Any]] = []
    for document in sorted(documents, key=lambda item: item.doc_id):
        flags = get_text_quality_flags(document.raw)
        rows.append(
            {
                "doc_id": document.doc_id,
                "title": document.title,
                "category": document.category,
                "format": document.format,
                "raw_length": len(document.raw),
                "line_count": document.raw.count("\n") + 1,
                "has_html": flags.has_html,
                "has_markdown": flags.has_markdown,
                "mojibake_markers": ", ".join(flags.mojibake_markers),
                "suspicious_sequences": ", ".join(_format_sequence(item) for item in flags.suspicious_sequences),
                "preview": _preview(document.raw),
            }
        )
    return _make_table(
        rows,
        (
            "doc_id",
            "title",
            "category",
            "format",
            "raw_length",
            "line_count",
            "has_html",
            "has_markdown",
            "mojibake_markers",
            "suspicious_sequences",
            "preview",
        ),
    )


def build_distribution_table(document_profile: AuditTable, column: str) -> AuditTable:
    """Считает распределение документов по указанной колонке.

    Аргументы:
        document_profile: Таблица профиля документов.
        column: Имя колонки для группировки.

    Возвращает:
        Таблицу ``значение → количество → доля``.
    """

    counts: dict[Any, int] = {}
    for row in document_profile.rows:
        counts[row[column]] = counts.get(row[column], 0) + 1
    total = len(document_profile)
    rows = [
        {column: value, "count": count, "share": round(count / total, 4) if total else 0.0}
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))
    ]
    return _make_table(rows, (column, "count", "share"))


def build_length_statistics(document_profile: AuditTable) -> AuditTable:
    """Считает статистики длины поля ``raw``.

    Аргументы:
        document_profile: Таблица профиля документов.

    Возвращает:
        Таблицу с минимумом, квантилями, средним и максимумом.
    """

    lengths = sorted(int(row["raw_length"]) for row in document_profile.rows)
    rows = [
        {"metric": "count", "value": len(lengths)},
        {"metric": "min", "value": min(lengths) if lengths else 0},
        {"metric": "p05", "value": _quantile(lengths, 0.05)},
        {"metric": "p25", "value": _quantile(lengths, 0.25)},
        {"metric": "median", "value": _quantile(lengths, 0.50)},
        {"metric": "mean", "value": round(sum(lengths) / len(lengths), 2) if lengths else 0},
        {"metric": "p75", "value": _quantile(lengths, 0.75)},
        {"metric": "p95", "value": _quantile(lengths, 0.95)},
        {"metric": "max", "value": max(lengths) if lengths else 0},
    ]
    return _make_table(rows, ("metric", "value"))


def find_duplicate_groups(
    documents: Sequence[KnowledgeDocument],
    *,
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> list[DuplicateGroup]:
    """Находит полные и почти полные дубли документов.

    Аргументы:
        documents: Документы базы знаний.
        near_duplicate_threshold: Порог похожести для почти полных дублей.

    Возвращает:
        Список групп дублей. Пустой список означает, что дубли не найдены.
    """

    normalized_by_id = {document.doc_id: normalize_text_for_duplicate_detection(document.raw) for document in documents}
    title_by_id = {document.doc_id: document.title for document in documents}
    parent = {document.doc_id: document.doc_id for document in documents}
    group_similarity: dict[int, float] = {}
    group_type: dict[int, str] = {}

    for left_index, left_document in enumerate(documents):
        left_text = normalized_by_id[left_document.doc_id]
        for right_document in documents[left_index + 1 :]:
            right_text = normalized_by_id[right_document.doc_id]
            if not _lengths_are_comparable(left_text, right_text):
                continue
            if left_text == right_text and left_text:
                similarity = 1.0
                match_type = "exact"
            else:
                if not _passes_duplicate_prefilter(left_text, right_text, near_duplicate_threshold):
                    continue
                similarity = text_similarity(left_text, right_text)
                match_type = "near"
            if similarity >= near_duplicate_threshold:
                root = _union(parent, left_document.doc_id, right_document.doc_id)
                previous_similarity = group_similarity.get(root, 1.0)
                group_similarity[root] = min(previous_similarity, similarity)
                if match_type == "exact" and group_type.get(root) != "near":
                    group_type[root] = "exact"
                else:
                    group_type[root] = "near"

    grouped_doc_ids: dict[int, list[int]] = {}
    for document in documents:
        root = _find(parent, document.doc_id)
        grouped_doc_ids.setdefault(root, []).append(document.doc_id)

    duplicate_groups: list[DuplicateGroup] = []
    for root, doc_ids in grouped_doc_ids.items():
        if len(doc_ids) < 2:
            continue
        sorted_doc_ids = tuple(sorted(doc_ids))
        duplicate_groups.append(
            DuplicateGroup(
                doc_ids=sorted_doc_ids,
                titles=tuple(title_by_id[doc_id] for doc_id in sorted_doc_ids),
                similarity=round(group_similarity.get(root, 1.0), 4),
                match_type=group_type.get(root, "near"),
            )
        )
    return sorted(duplicate_groups, key=lambda group: (group.match_type, group.doc_ids))


def duplicate_groups_to_dataframe(duplicate_groups: Sequence[DuplicateGroup]) -> AuditTable:
    """Преобразует группы дублей в таблицу для notebook.

    Название сохранено для удобства notebook, хотя функция возвращает
    ``AuditTable``, а не pandas ``DataFrame``.

    Аргументы:
        duplicate_groups: Группы дублей из ``audit_corpus``.

    Возвращает:
        Таблицу с идентификаторами, заголовками и типом совпадения.
    """

    return _make_table(
        [
            {
                "doc_ids": ", ".join(str(doc_id) for doc_id in group.doc_ids),
                "titles": " | ".join(group.titles),
                "similarity": group.similarity,
                "match_type": group.match_type,
            }
            for group in duplicate_groups
        ],
        ("doc_ids", "titles", "similarity", "match_type"),
    )


def build_problem_decisions(
    *,
    html_count: int,
    markdown_count: int,
    mojibake_count: int,
    suspicious_count: int,
    duplicate_group_count: int,
    short_count: int,
    long_count: int,
) -> AuditTable:
    """Формирует таблицу решений по найденным проблемам аудита.

    Аргументы:
        html_count: Количество документов с HTML-признаками.
        markdown_count: Количество документов с Markdown-признаками.
        mojibake_count: Количество документов с mojibake-маркерами.
        suspicious_count: Количество документов с подозрительными фрагментами.
        duplicate_group_count: Количество групп дублей.
        short_count: Количество коротких документов.
        long_count: Количество длинных документов.

    Возвращает:
        Таблицу решений для отчётной части notebook.
    """

    return _make_table(
        [
            {
                "problem": "HTML-разметка",
                "count": html_count,
                "decision": "исправлять на этапе 1.3: извлечь читаемый текст без потери ссылок и смысловых блоков",
            },
            {
                "problem": "Markdown-разметка",
                "count": markdown_count,
                "decision": "исправлять на этапе 1.3: убрать служебные символы Markdown, сохранив структуру списков и заголовков",
            },
            {
                "problem": "mojibake/ошибки кодировки",
                "count": mojibake_count,
                "decision": "исправлять только при уверенной эвристике; спорные фрагменты помечать для ручной проверки",
            },
            {
                "problem": "мусорные символы и подозрительные последовательности",
                "count": suspicious_count,
                "decision": "исправлять безопасные случаи нормализацией пробелов и удалением управляющих символов; остальное оставить до ручной проверки",
            },
            {
                "problem": "короткие документы",
                "count": short_count,
                "decision": "не удалять автоматически; проверить, являются ли это заглушки, объявления или полезные короткие статьи",
            },
            {
                "problem": "длинные документы",
                "count": long_count,
                "decision": "оставить в корпусе и учитывать при выборе стратегии чанкинга",
            },
            {
                "problem": "полные или почти полные дубли",
                "count": duplicate_group_count,
                "decision": "фильтровать или объединять только после ручной проверки совпадающих групп",
            },
        ],
        ("problem", "count", "decision"),
    )


def print_audit_summary(audit: CorpusAudit) -> None:
    """Печатает краткую текстовую сводку аудита.

    Аргументы:
        audit: Результат ``audit_corpus``.
    """

    print(f"Документов: {audit.total_documents}")
    print(f"Категорий: {len(audit.category_distribution)}")
    print(f"Форматов: {len(audit.format_distribution)}")
    print(f"Коротких документов: {len(audit.short_documents)}")
    print(f"Длинных документов: {len(audit.long_documents)}")
    print(f"Документов с HTML: {len(audit.html_documents)}")
    print(f"Документов с Markdown: {len(audit.markdown_documents)}")
    print(f"Документов с mojibake: {len(audit.mojibake_documents)}")
    print(f"Документов с подозрительными фрагментами: {len(audit.suspicious_documents)}")
    print(f"Групп дублей: {len(audit.duplicate_groups)}")


def _make_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> AuditTable:
    """Создаёт ``AuditTable`` с фиксированным порядком колонок."""

    ordered_rows = tuple({column: row.get(column, "") for column in columns} for row in rows)
    return AuditTable(rows=ordered_rows, columns=tuple(columns))


def _filter_table(table: AuditTable, predicate: Any) -> AuditTable:
    """Фильтрует строки таблицы по предикату."""

    return _make_table([row for row in table.rows if predicate(row)], table.columns)


def _quantile(values: Sequence[int], quantile: float) -> int:
    """Считает ближайший квантиль без зависимости от ``numpy``.

    Аргументы:
        values: Набор целых значений.
        quantile: Квантиль от ``0`` до ``1``.

    Возвращает:
        Значение ближайшего элемента отсортированного списка.
    """

    if not values:
        return 0
    if quantile < 0 or quantile > 1:
        raise ValueError(f"Квантиль должен быть в диапазоне [0, 1], получено: {quantile}.")
    sorted_values = sorted(values)
    index = round((len(sorted_values) - 1) * quantile)
    return sorted_values[index]


def _lengths_are_comparable(left: str, right: str) -> bool:
    """Проверяет, имеет ли смысл сравнивать два текста как почти дубли.

    Аргументы:
        left: Первый нормализованный текст.
        right: Второй нормализованный текст.

    Возвращает:
        ``True``, если длины текстов сопоставимы.
    """

    shorter = min(len(left), len(right))
    longer = max(len(left), len(right))
    if longer == 0:
        return True
    return shorter / longer >= 0.8


def _passes_duplicate_prefilter(left: str, right: str, threshold: float) -> bool:
    """Быстро отсеивает пары, которые не похожи на почти полные дубли.

    Аргументы:
        left: Первый нормализованный текст.
        right: Второй нормализованный текст.
        threshold: Итоговый порог похожести для почти дублей.

    Возвращает:
        ``True``, если пару стоит проверять дорогим точным сравнением.
    """

    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return False
    token_overlap = len(left_tokens.intersection(right_tokens)) / max(len(left_tokens), len(right_tokens))
    if token_overlap < 0.75:
        return False

    matcher = SequenceMatcher(a=left, b=right, autojunk=False)
    return matcher.real_quick_ratio() >= threshold and matcher.quick_ratio() >= threshold


def _find(parent: dict[int, int], doc_id: int) -> int:
    """Находит корень множества в структуре union-find."""

    if parent[doc_id] != doc_id:
        parent[doc_id] = _find(parent, parent[doc_id])
    return parent[doc_id]


def _union(parent: dict[int, int], left_doc_id: int, right_doc_id: int) -> int:
    """Объединяет два множества в структуре union-find."""

    left_root = _find(parent, left_doc_id)
    right_root = _find(parent, right_doc_id)
    if left_root == right_root:
        return left_root
    new_root = min(left_root, right_root)
    old_root = max(left_root, right_root)
    parent[old_root] = new_root
    return new_root


def _preview(text: str, *, max_length: int = 180) -> str:
    """Готовит короткий однострочный фрагмент текста для таблиц."""

    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 1].rstrip() + "…"


def _format_sequence(value: str) -> str:
    """Форматирует подозрительный фрагмент для вывода в таблицу."""

    escaped = value.encode("unicode_escape").decode("ascii")
    if len(escaped) > 40:
        return escaped[:37] + "..."
    return escaped


def _stringify(value: Any) -> str:
    """Преобразует значение ячейки в строку для табличного вывода."""

    return str(value)


def _escape_html(value: str) -> str:
    """Экранирует значение для HTML-представления таблицы."""

    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


__all__ = [
    "AuditTable",
    "CorpusAudit",
    "DEFAULT_LONG_QUANTILE",
    "DEFAULT_NEAR_DUPLICATE_THRESHOLD",
    "DEFAULT_SHORT_QUANTILE",
    "DuplicateGroup",
    "audit_corpus",
    "build_distribution_table",
    "build_document_profile",
    "build_length_statistics",
    "build_problem_decisions",
    "duplicate_groups_to_dataframe",
    "find_duplicate_groups",
    "print_audit_summary",
]
