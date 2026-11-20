"""Вспомогательные функции конфигурации прототипа RAG-ассистента «Фильмоскоп».

Модуль централизует runtime-настройки, которые должны приходить из переменных
окружения или локального файла ``.env``. Он намеренно не зависит от остальных
частей проекта, чтобы его можно было импортировать на раннем этапе из notebook
и скриптов.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class RuntimeSettings:
    """Общие настройки воспроизводимости и запуска.

    Атрибуты:
        random_seed: Seed для воспроизводимых экспериментов.
    """

    random_seed: int


@dataclass(frozen=True)
class DataSettings:
    """Настройки источников данных проекта.

    Атрибуты:
        dataset_path: Путь к JSON-файлу базы знаний.
    """

    dataset_path: Path


@dataclass(frozen=True)
class LlmSettings:
    """Настройки chat-модели для генерации ответов.

    Атрибуты:
        base_url: OpenAI-compatible base URL для генератора ответов.
        api_key: Значение API key для генератора ответов.
        model_name: Имя chat-модели для генератора ответов.
        temperature: Температура сэмплирования для генератора ответов.
        timeout_seconds: Таймаут запроса для генерации ответов.
        max_tokens: Максимальное число генерируемых токенов для ответов.
    """

    base_url: str
    api_key: str
    model_name: str
    temperature: float
    timeout_seconds: int
    max_tokens: int


@dataclass(frozen=True)
class JudgeSettings:
    """Настройки chat-модели для автоматической оценки ответов.

    Атрибуты:
        base_url: OpenAI-compatible base URL для judge-модели.
        api_key: Значение API key для judge-модели.
        model_name: Имя chat-модели для оценки ответов.
        temperature: Температура сэмплирования для оценки ответов.
        timeout_seconds: Таймаут запроса для оценки ответов.
        max_tokens: Максимальное число генерируемых токенов для оценки.
    """

    base_url: str
    api_key: str
    model_name: str
    temperature: float
    timeout_seconds: int
    max_tokens: int


@dataclass(frozen=True)
class EmbeddingSettings:
    """Настройки embedding-модели.

    Атрибуты:
        model_name: Имя embedding-модели.
        device: Селектор устройства для расчёта embeddings.
        batch_size: Размер batch для inference embedding-модели.
        normalize: Нужно ли нормализовать embedding-векторы.
    """

    model_name: str
    device: str
    batch_size: int
    normalize: bool


@dataclass(frozen=True)
class ChromaSettings:
    """Настройки локального хранилища Chroma.

    Атрибуты:
        persist_dir: Локальная директория персистентности Chroma.
        collection_prefix: Префикс имён коллекций Chroma.
    """

    persist_dir: Path
    collection_prefix: str


@dataclass(frozen=True)
class ChunkingSettings:
    """Настройки стратегий чанкинга документов.

    Атрибуты:
        chunk_size: Размер чанка по умолчанию для основной стратегии чанкинга.
        chunk_overlap: Перекрытие чанков по умолчанию для основной стратегии.
        alt_chunk_size: Размер чанка для альтернативной стратегии чанкинга.
        alt_chunk_overlap: Перекрытие чанков для альтернативной стратегии.
    """

    chunk_size: int
    chunk_overlap: int
    alt_chunk_size: int
    alt_chunk_overlap: int


@dataclass(frozen=True)
class RetrievalSettings:
    """Настройки retrieval-этапа.

    Атрибуты:
        top_k: Количество итоговых retrieval-результатов.
        fetch_k: Количество кандидатов до фильтрации или reranking.
        score_threshold: Минимальный допустимый retrieval score.
        bm25_enabled: Включён ли BM25 retrieval.
        rrf_enabled: Включён ли reciprocal rank fusion.
        rrf_k: Константа сглаживания RRF.
    """

    top_k: int
    fetch_k: int
    score_threshold: float
    bm25_enabled: bool
    rrf_enabled: bool
    rrf_k: int


@dataclass(frozen=True)
class RerankerSettings:
    """Настройки reranker-модели.

    Атрибуты:
        enabled: Включён ли reranking.
        model_name: Имя reranker-модели.
        top_n: Количество кандидатов, передаваемых в reranker.
        device: Селектор устройства для reranker.
    """

    enabled: bool
    model_name: str
    top_n: int
    device: str


@dataclass(frozen=True)
class GenerationSettings:
    """Настройки постобработки и политики генерации ответов.

    Атрибуты:
        refusal_phrase: Стандартная фраза отказа для неподдержанных вопросов.
    """

    refusal_phrase: str


@dataclass(frozen=True, init=False)
class Settings:
    """Сгруппированные настройки прототипа RAG-системы «Фильмоскоп».

    Атрибуты:
        runtime: Общие настройки воспроизводимости и запуска.
        data: Настройки источников данных проекта.
        llm: Настройки chat-модели для генерации ответов.
        judge: Настройки chat-модели для автоматической оценки ответов.
        embedding: Настройки embedding-модели.
        chroma: Настройки локального хранилища Chroma.
        chunking: Настройки стратегий чанкинга документов.
        retrieval: Настройки retrieval-этапа.
        reranker: Настройки reranker-модели.
        generation: Настройки постобработки и политики генерации ответов.
    """

    __runtime: RuntimeSettings
    __data: DataSettings
    __llm: LlmSettings
    __judge: JudgeSettings
    __embedding: EmbeddingSettings
    __chroma: ChromaSettings
    __chunking: ChunkingSettings
    __retrieval: RetrievalSettings
    __reranker: RerankerSettings
    __generation: GenerationSettings

    def __init__(
        self,
        *,
        runtime: RuntimeSettings,
        data: DataSettings,
        llm: LlmSettings,
        judge: JudgeSettings,
        embedding: EmbeddingSettings,
        chroma: ChromaSettings,
        chunking: ChunkingSettings,
        retrieval: RetrievalSettings,
        reranker: RerankerSettings,
        generation: GenerationSettings,
    ) -> None:
        """Инициализирует сгруппированные настройки проекта.

        Аргументы:
            runtime: Общие настройки воспроизводимости и запуска.
            data: Настройки источников данных проекта.
            llm: Настройки chat-модели для генерации ответов.
            judge: Настройки chat-модели для автоматической оценки ответов.
            embedding: Настройки embedding-модели.
            chroma: Настройки локального хранилища Chroma.
            chunking: Настройки стратегий чанкинга документов.
            retrieval: Настройки retrieval-этапа.
            reranker: Настройки reranker-модели.
            generation: Настройки постобработки и политики генерации ответов.
        """

        object.__setattr__(self, "_Settings__runtime", runtime)
        object.__setattr__(self, "_Settings__data", data)
        object.__setattr__(self, "_Settings__llm", llm)
        object.__setattr__(self, "_Settings__judge", judge)
        object.__setattr__(self, "_Settings__embedding", embedding)
        object.__setattr__(self, "_Settings__chroma", chroma)
        object.__setattr__(self, "_Settings__chunking", chunking)
        object.__setattr__(self, "_Settings__retrieval", retrieval)
        object.__setattr__(self, "_Settings__reranker", reranker)
        object.__setattr__(self, "_Settings__generation", generation)

    @property
    def runtime(self) -> RuntimeSettings:
        """Общие настройки воспроизводимости и запуска."""

        return self.__runtime

    @property
    def data(self) -> DataSettings:
        """Настройки источников данных проекта."""

        return self.__data

    @property
    def llm(self) -> LlmSettings:
        """Настройки chat-модели для генерации ответов."""

        return self.__llm

    @property
    def judge(self) -> JudgeSettings:
        """Настройки chat-модели для автоматической оценки ответов."""

        return self.__judge

    @property
    def embedding(self) -> EmbeddingSettings:
        """Настройки embedding-модели."""

        return self.__embedding

    @property
    def chroma(self) -> ChromaSettings:
        """Настройки локального хранилища Chroma."""

        return self.__chroma

    @property
    def chunking(self) -> ChunkingSettings:
        """Настройки стратегий чанкинга документов."""

        return self.__chunking

    @property
    def retrieval(self) -> RetrievalSettings:
        """Настройки retrieval-этапа."""

        return self.__retrieval

    @property
    def reranker(self) -> RerankerSettings:
        """Настройки reranker-модели."""

        return self.__reranker

    @property
    def generation(self) -> GenerationSettings:
        """Настройки постобработки и политики генерации ответов."""

        return self.__generation


def load_settings(env_path: str | Path | None = DEFAULT_ENV_PATH, *, override: bool = False) -> Settings:
    """Загружает настройки проекта из ``.env`` и переменных окружения.

    Аргументы:
        env_path: Путь к dotenv-файлу. Если передано ``None``, dotenv-файл не
            загружается.
        override: Должны ли значения из dotenv-файла перезаписывать уже заданные
            переменные окружения.

    Возвращает:
        Распарсенные настройки проекта.
    """

    if env_path is not None:
        load_env_file(Path(env_path), override=override)

    return Settings(
        runtime=RuntimeSettings(random_seed=_get_int("RANDOM_SEED", 42)),
        data=DataSettings(dataset_path=_get_path("DATASET_PATH", "filmoskop_kb.json")),
        llm=LlmSettings(
            base_url=_get_str("LLM_BASE_URL", "http://localhost:8080/v1"),
            api_key=_get_str("LLM_API_KEY", "not-needed-for-local-server"),
            model_name=_get_str("LLM_MODEL_NAME", "unsloth/Qwen3.5"),
            temperature=_get_float("LLM_TEMPERATURE", 0.0),
            timeout_seconds=_get_int("LLM_TIMEOUT_SECONDS", 120),
            max_tokens=_get_int("LLM_MAX_TOKENS", 1024),
        ),
        judge=JudgeSettings(
            base_url=_get_str("JUDGE_BASE_URL", "http://localhost:8080/v1"),
            api_key=_get_str("JUDGE_API_KEY", "not-needed-for-local-server"),
            model_name=_get_str("JUDGE_MODEL_NAME", "unsloth/Qwen3.5"),
            temperature=_get_float("JUDGE_TEMPERATURE", 0.0),
            timeout_seconds=_get_int("JUDGE_TIMEOUT_SECONDS", 120),
            max_tokens=_get_int("JUDGE_MAX_TOKENS", 1024),
        ),
        embedding=EmbeddingSettings(
            model_name=_get_str("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base"),
            device=_get_str("EMBEDDING_DEVICE", "auto"),
            batch_size=_get_int("EMBEDDING_BATCH_SIZE", 32),
            normalize=_get_bool("NORMALIZE_EMBEDDINGS", True),
        ),
        chroma=ChromaSettings(
            persist_dir=_get_path("CHROMA_PERSIST_DIR", ".chroma"),
            collection_prefix=_get_str("CHROMA_COLLECTION_PREFIX", "filmoskop"),
        ),
        chunking=ChunkingSettings(
            chunk_size=_get_int("CHUNK_SIZE", 600),
            chunk_overlap=_get_int("CHUNK_OVERLAP", 100),
            alt_chunk_size=_get_int("ALT_CHUNK_SIZE", 450),
            alt_chunk_overlap=_get_int("ALT_CHUNK_OVERLAP", 80),
        ),
        retrieval=RetrievalSettings(
            top_k=_get_int("RETRIEVAL_TOP_K", 5),
            fetch_k=_get_int("RETRIEVAL_FETCH_K", 20),
            score_threshold=_get_float("RETRIEVAL_SCORE_THRESHOLD", 0.0),
            bm25_enabled=_get_bool("BM25_ENABLED", True),
            rrf_enabled=_get_bool("RRF_ENABLED", True),
            rrf_k=_get_int("RRF_K", 30),
        ),
        reranker=RerankerSettings(
            enabled=_get_bool("RERANKER_ENABLED", False),
            model_name=_get_str("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3"),
            top_n=_get_int("RERANKER_TOP_N", 20),
            device=_get_str("RERANKER_DEVICE", "auto"),
        ),
        generation=GenerationSettings(
            refusal_phrase=_get_str(
                "REFUSAL_PHRASE",
                "В базе знаний нет информации, чтобы надёжно ответить на этот вопрос.",
            )
        ),
    )


def load_env_file(env_path: Path = DEFAULT_ENV_PATH, *, override: bool = False) -> None:
    """Загружает переменные из dotenv-файла в ``os.environ``.

    Аргументы:
        env_path: Путь к dotenv-файлу.
        override: Должны ли значения из dotenv-файла перезаписывать уже заданные
            переменные окружения.
    """

    if not env_path.exists():
        return

    from dotenv import load_dotenv

    load_dotenv(dotenv_path=env_path, override=override)


def _get_str(name: str, default: str) -> str:
    """Читает строковую переменную окружения.

    Аргументы:
        name: Имя переменной окружения.
        default: Значение по умолчанию, если переменная отсутствует или пуста.

    Возвращает:
        Значение переменной окружения или значение по умолчанию.
    """

    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _get_int(name: str, default: int) -> int:
    """Читает целочисленную переменную окружения.

    Аргументы:
        name: Имя переменной окружения.
        default: Значение по умолчанию, если переменная отсутствует или пуста.

    Возвращает:
        Распарсенное целочисленное значение.

    Исключения:
        ValueError: Если заданное значение нельзя распарсить как целое число.
    """

    value = _get_str(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Переменная окружения {name} должна быть целым числом, получено {value!r}.") from exc


def _get_float(name: str, default: float) -> float:
    """Читает вещественную переменную окружения.

    Аргументы:
        name: Имя переменной окружения.
        default: Значение по умолчанию, если переменная отсутствует или пуста.

    Возвращает:
        Распарсенное вещественное значение.

    Исключения:
        ValueError: Если заданное значение нельзя распарсить как вещественное число.
    """

    value = _get_str(name, str(default))
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Переменная окружения {name} должна быть вещественным числом, получено {value!r}.") from exc


def _get_bool(name: str, default: bool) -> bool:
    """Читает булеву переменную окружения.

    Аргументы:
        name: Имя переменной окружения.
        default: Значение по умолчанию, если переменная отсутствует или пуста.

    Возвращает:
        Распарсенное булево значение.

    Исключения:
        ValueError: Если заданное значение не является поддерживаемым булевым литералом.
    """

    value = _get_str(name, str(default)).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Переменная окружения {name} должна быть булевым значением, получено {value!r}.")


def _get_path(name: str, default: str) -> Path:
    """Читает переменную окружения с путём относительно корня проекта.

    Аргументы:
        name: Имя переменной окружения.
        default: Путь по умолчанию, если переменная отсутствует или пуста.

    Возвращает:
        Абсолютный путь, если настроенный путь относителен корню проекта; иначе
        настроенный абсолютный путь.
    """

    path = Path(_get_str(name, default)).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


__all__ = [
    "ChromaSettings",
    "ChunkingSettings",
    "DEFAULT_ENV_PATH",
    "DataSettings",
    "EmbeddingSettings",
    "GenerationSettings",
    "JudgeSettings",
    "LlmSettings",
    "PROJECT_ROOT",
    "RerankerSettings",
    "RetrievalSettings",
    "RuntimeSettings",
    "Settings",
    "load_env_file",
    "load_settings",
]