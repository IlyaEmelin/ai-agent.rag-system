"""Wrapper embedding-модели для retrieval-пайплайна RAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.documents import Document

from src.config import EmbeddingSettings


DEFAULT_EMBEDDING_QUERY_PREFIX = "query: "
DEFAULT_EMBEDDING_PASSAGE_PREFIX = "passage: "


@dataclass(frozen=True)
class EmbeddingModelInfo:
    """Сводная информация о загруженной embedding-модели.

    Атрибуты:
        model_name: Имя модели из настроек проекта.
        device: Фактическое устройство, выбранное для inference.
        batch_size: Размер batch для построения embeddings.
        normalize: Нормализуются ли выходные векторы.
        dimension: Размерность embedding-вектора, если её удалось определить.
    """

    model_name: str
    device: str
    batch_size: int
    normalize: bool
    dimension: int | None


class SentenceTransformerEmbeddingModel:
    """Локальный wrapper над ``sentence-transformers`` для RAG embeddings.

    Класс намеренно импортирует ``sentence_transformers`` лениво при создании
    экземпляра. Это позволяет импортировать модуль проекта даже в окружении, где
    зависимости ещё не установлены, и получить понятную ошибку только на этапе
    загрузки embedding-модели.
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        """Загружает embedding-модель по настройкам проекта.

        Аргументы:
            settings: Блок ``settings.embedding`` из ``src.config``.

        Исключения:
            ImportError: Если пакет ``sentence-transformers`` не установлен.
            ValueError: Если настройки embedding-модели некорректны.
        """

        validate_embedding_settings(settings)
        self.model_name = settings.model_name
        self.batch_size = int(settings.batch_size)
        self.normalize = bool(settings.normalize)
        self.device = resolve_embedding_device(settings.device)
        self._model = self._load_model()
        self.dimension = self._resolve_dimension()

    @property
    def info(self) -> EmbeddingModelInfo:
        """Возвращает сведения о загруженной embedding-модели."""

        return EmbeddingModelInfo(
            model_name=self.model_name,
            device=self.device,
            batch_size=self.batch_size,
            normalize=self.normalize,
            dimension=self.dimension,
        )

    def embed_texts(self, texts: Sequence[str], *, prefix: str = DEFAULT_EMBEDDING_PASSAGE_PREFIX) -> list[list[float]]:
        """Строит embeddings для произвольного списка текстов.

        Аргументы:
            texts: Последовательность непустых текстов.
            prefix: Префикс инструкции для E5-подобных моделей. Для документов
                используется ``passage: ``, для вопросов — ``query: ``.

        Возвращает:
            Список embedding-векторов в том же порядке, что и входные тексты.

        Исключения:
            ValueError: Если входные тексты пустые или содержат нестроковые
                значения.
        """

        normalized_texts = prepare_embedding_texts(texts, prefix=prefix)
        if not normalized_texts:
            return []
        embeddings = self._model.encode(
            normalized_texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_documents(self, documents: Sequence[Document]) -> list[list[float]]:
        """Строит passage embeddings для чанков LangChain ``Document``.

        Аргументы:
            documents: Последовательность документов или чанков.

        Возвращает:
            Список embedding-векторов для ``page_content`` каждого документа.
        """

        texts = [document.page_content for document in documents]
        return self.embed_texts(texts, prefix=DEFAULT_EMBEDDING_PASSAGE_PREFIX)

    def embed_query(self, query: str) -> list[float]:
        """Строит query embedding для пользовательского вопроса.

        Аргументы:
            query: Текст вопроса.

        Возвращает:
            Один embedding-вектор.
        """

        return self.embed_texts([query], prefix=DEFAULT_EMBEDDING_QUERY_PREFIX)[0]

    def _load_model(self) -> Any:
        """Загружает ``SentenceTransformer`` с понятной ошибкой окружения."""

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Пакет sentence-transformers не установлен в текущем окружении. "
                "Установите зависимости командой: pip install -r requirements.txt"
            ) from exc

        return SentenceTransformer(self.model_name, device=self.device)

    def _resolve_dimension(self) -> int | None:
        """Определяет размерность embedding-вектора, если модель её сообщает."""

        dimension_getter = getattr(self._model, "get_embedding_dimension", None)
        if dimension_getter is None:
            dimension_getter = getattr(self._model, "get_sentence_embedding_dimension", None)
        if dimension_getter is None:
            return None
        dimension = dimension_getter()
        return int(dimension) if dimension is not None else None


def build_embedding_model(settings: EmbeddingSettings) -> SentenceTransformerEmbeddingModel:
    """Создаёт wrapper embedding-модели из настроек проекта.

    Аргументы:
        settings: Блок ``settings.embedding`` из ``src.config``.

    Возвращает:
        Загруженный wrapper embedding-модели.
    """

    return SentenceTransformerEmbeddingModel(settings)


def prepare_embedding_texts(texts: Sequence[str], *, prefix: str) -> list[str]:
    """Нормализует тексты перед подачей в embedding-модель.

    Аргументы:
        texts: Последовательность текстов.
        prefix: Префикс, который добавляется к каждому тексту.

    Возвращает:
        Список текстов с E5-инструкцией.

    Исключения:
        ValueError: Если найден пустой или нестроковый текст.
    """

    prepared_texts: list[str] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError(f"Текст для embedding должен быть строкой, индекс={index}, получено: {type(text)!r}.")
        stripped_text = text.strip()
        if not stripped_text:
            raise ValueError(f"Текст для embedding не должен быть пустым, индекс={index}.")
        prepared_texts.append(f"{prefix}{stripped_text}")
    return prepared_texts


def resolve_embedding_device(device: str) -> str:
    """Преобразует селектор устройства из настроек в значение для модели.

    Аргументы:
        device: Значение ``EMBEDDING_DEVICE``. Поддерживается ``auto``.

    Возвращает:
        ``cuda`` при доступной CUDA и ``auto``, иначе ``cpu`` или явно заданное
        пользователем устройство.
    """

    normalized_device = device.strip().lower()
    if not normalized_device:
        raise ValueError("EMBEDDING_DEVICE не должен быть пустым.")
    if normalized_device != "auto":
        return normalized_device

    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_embedding_settings(settings: EmbeddingSettings) -> None:
    """Проверяет настройки embedding-модели.

    Аргументы:
        settings: Блок ``settings.embedding`` из ``src.config``.

    Исключения:
        ValueError: Если настройки некорректны.
    """

    if not isinstance(settings.model_name, str) or not settings.model_name.strip():
        raise ValueError(f"Имя embedding-модели должно быть непустой строкой, получено: {settings.model_name!r}.")
    if not isinstance(settings.device, str) or not settings.device.strip():
        raise ValueError(f"Устройство embedding-модели должно быть непустой строкой, получено: {settings.device!r}.")
    if int(settings.batch_size) <= 0:
        raise ValueError(f"EMBEDDING_BATCH_SIZE должен быть положительным числом, получено: {settings.batch_size}.")
    if not isinstance(settings.normalize, bool):
        raise ValueError(f"NORMALIZE_EMBEDDINGS должен быть булевым значением, получено: {settings.normalize!r}.")


def summarize_embeddings(embeddings: Sequence[Sequence[float]]) -> dict[str, int | float]:
    """Собирает краткую сводку по построенным embedding-векторам.

    Аргументы:
        embeddings: Последовательность embedding-векторов.

    Возвращает:
        Словарь с количеством векторов и размерностью.
    """

    if not embeddings:
        return {"embeddings": 0, "dimension": 0, "min_norm": 0.0, "max_norm": 0.0}

    dimensions = {len(vector) for vector in embeddings}
    if len(dimensions) != 1:
        raise ValueError(f"Все embedding-векторы должны иметь одинаковую размерность, получено: {sorted(dimensions)}.")

    norms = [_l2_norm(vector) for vector in embeddings]
    return {
        "embeddings": len(embeddings),
        "dimension": dimensions.pop(),
        "min_norm": round(min(norms), 6),
        "max_norm": round(max(norms), 6),
    }


def _l2_norm(vector: Sequence[float]) -> float:
    """Вычисляет L2-норму одного embedding-вектора."""

    return sum(float(value) ** 2 for value in vector) ** 0.5


__all__ = [
    "DEFAULT_EMBEDDING_PASSAGE_PREFIX",
    "DEFAULT_EMBEDDING_QUERY_PREFIX",
    "EmbeddingModelInfo",
    "SentenceTransformerEmbeddingModel",
    "build_embedding_model",
    "prepare_embedding_texts",
    "resolve_embedding_device",
    "summarize_embeddings",
    "validate_embedding_settings",
]