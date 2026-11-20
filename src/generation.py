"""Вспомогательные функции генерации ответов через локальную LLM для RAG-ассистента."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from src.config import GenerationSettings, LlmSettings, RetrievalSettings, Settings
from src.retrieval import BM25Index, RetrievedChunk, hybrid_retrieve


DEFAULT_CONTEXT_CHARS_PER_CHUNK = 1400
DEFAULT_CONTEXT_SEPARATOR = "\n\n---\n\n"
QWEN_DISABLE_THINKING_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


@dataclass(frozen=True)
class LlmResponse:
    """Ответ chat-модели и служебная информация о вызове.

    Атрибуты:
        content: Финальный текст ответа из ``choices[0].message.content``.
        model_name: Имя модели, вернувшей ответ.
        usage: Статистика токенов, если сервер её вернул.
        raw_response: Исходный объект ответа OpenAI-compatible клиента.
    """

    content: str
    model_name: str
    usage: dict[str, Any]
    raw_response: Any


@dataclass(frozen=True)
class RagResponse:
    """Результат полного RAG-пайплайна для одного пользовательского вопроса.

    Атрибуты:
        question: Исходный вопрос пользователя.
        answer: Ответ генератора или согласованная фраза отказа.
        retrieved_chunks: Чанки, переданные в контекст генератора.
        sources: Человекочитаемые источники ответа.
        prompt: Финальный prompt, отправленный в LLM.
        diagnostics: Служебная диагностика retrieval и generation этапов.
    """

    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    sources: list[dict[str, Any]]
    prompt: str
    diagnostics: dict[str, Any]


class OpenAIChatClient:
    """Минимальный OpenAI-compatible клиент для локального Qwen-сервера.

    Класс использует официальный пакет ``openai`` напрямую, чтобы явно передать
    дополнительное поле payload ``chat_template_kwargs.enable_thinking=false`` и
    затем проверить, что финальный ответ пришёл именно в
    ``choices[0].message.content``.
    """

    def __init__(self, settings: LlmSettings) -> None:
        """Создаёт клиента по настройкам ``settings.llm``.

        Аргументы:
            settings: Настройки LLM из ``src.config``.

        Исключения:
            ImportError: Если пакет ``openai`` не установлен.
            ValueError: Если настройки клиента некорректны.
        """

        validate_llm_settings(settings)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Пакет openai не установлен. Установите зависимости: pip install -r requirements.txt") from exc

        self.settings = settings
        self._client = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.timeout_seconds,
        )

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> LlmResponse:
        """Отправляет prompt в chat-completion endpoint и возвращает content.

        Аргументы:
            prompt: Пользовательский prompt.
            system_prompt: Необязательное системное сообщение.

        Возвращает:
            ``LlmResponse`` с непустым ``content``.

        Исключения:
            RuntimeError: Если сервер недоступен, вернул ошибку или не положил
                финальный ответ в ``choices[0].message.content``.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt для LLM должен быть непустой строкой.")
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt.strip()})

        try:
            response = self._client.chat.completions.create(
                model=self.settings.model_name,
                messages=messages,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
                extra_body=QWEN_DISABLE_THINKING_BODY,
            )
        except Exception as exc:
            raise RuntimeError(
                "Не удалось получить ответ LLM. Проверьте, что локальный OpenAI-compatible "
                f"сервер доступен по {self.settings.base_url!r}, модель {self.settings.model_name!r} загружена, "
                "а endpoint /v1/chat/completions поддерживается."
            ) from exc

        content = extract_message_content(response)
        return LlmResponse(
            content=content,
            model_name=str(getattr(response, "model", self.settings.model_name) or self.settings.model_name),
            usage=_object_to_dict(getattr(response, "usage", None)),
            raw_response=response,
        )


def create_chat_client(settings: LlmSettings) -> OpenAIChatClient:
    """Создаёт OpenAI-compatible chat-клиент для генерации ответов.

    Аргументы:
        settings: Блок ``settings.llm``.

    Возвращает:
        Готовый ``OpenAIChatClient``.
    """

    return OpenAIChatClient(settings)


def create_langchain_chat_model(settings: LlmSettings) -> Any:
    """Создаёт ``langchain-openai.ChatOpenAI`` для локального Qwen-сервера.

    Функция оставлена как основной LangChain-совместимый вариант из плана
    проекта. В ``extra_body`` передаётся ``chat_template_kwargs`` с отключённым
    thinking-режимом, чтобы OpenAI-compatible сервер возвращал финальный текст в
    ``message.content``.

    Аргументы:
        settings: Блок ``settings.llm``.

    Возвращает:
        Экземпляр ``ChatOpenAI``.

    Исключения:
        ImportError: Если пакет ``langchain-openai`` не установлен.
        ValueError: Если настройки клиента некорректны.
    """

    validate_llm_settings(settings)
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "Пакет langchain-openai не установлен. Установите зависимости: pip install -r requirements.txt"
        ) from exc

    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=settings.temperature,
        timeout=settings.timeout_seconds,
        max_completion_tokens=settings.max_tokens,
        extra_body=QWEN_DISABLE_THINKING_BODY,
    )


def build_rag_context(
    chunks: Sequence[RetrievedChunk],
    *,
    max_chars_per_chunk: int = DEFAULT_CONTEXT_CHARS_PER_CHUNK,
) -> str:
    """Формирует текстовый контекст из top-K retrieval-чанков.

    Аргументы:
        chunks: Найденные retrieval-чанки в ранжированном порядке.
        max_chars_per_chunk: Максимальная длина текста одного чанка в prompt.

    Возвращает:
        Контекст с явными границами фрагментов, ``doc_id`` и названиями разделов.
    """

    if max_chars_per_chunk <= 0:
        raise ValueError(f"max_chars_per_chunk должен быть положительным, получено: {max_chars_per_chunk}.")
    context_blocks: list[str] = []
    for chunk in chunks:
        title = str(chunk.metadata.get("title", "")).strip()
        category = str(chunk.metadata.get("category", "")).strip()
        text = _trim_text(chunk.text, max_chars_per_chunk)
        context_blocks.append(
            "\n".join(
                [
                    f"[Источник {chunk.rank}]",
                    f"doc_id: {chunk.doc_id}",
                    f"chunk_id: {chunk.chunk_id}",
                    f"Название: {title}",
                    f"Категория: {category}",
                    "Фрагмент:",
                    text,
                ]
            )
        )
    return DEFAULT_CONTEXT_SEPARATOR.join(context_blocks)


def build_rag_prompt(question: str, chunks: Sequence[RetrievedChunk], refusal_phrase: str) -> str:
    """Строит prompt для заземлённого RAG-ответа.

    Prompt явно разделяет контекст и вопрос, запрещает отвечать вне контекста и
    фиксирует единую фразу отказа для случаев, когда в контексте нет ответа.

    Аргументы:
        question: Пользовательский вопрос.
        chunks: Retrieval-чанки, которые будут переданы модели как контекст.
        refusal_phrase: Согласованная фраза отказа из ``REFUSAL_PHRASE``.

    Возвращает:
        Готовый prompt для chat-модели.
    """

    normalized_question = _require_non_empty_string(question, "question")
    normalized_refusal = _require_non_empty_string(refusal_phrase, "refusal_phrase")
    context = build_rag_context(chunks) if chunks else "Контекст не найден."
    return f"""Ты — ассистент службы поддержки онлайн-кинотеатра «Фильмоскоп».

Отвечай только на основании блока «Контекст». Не используй внешние знания и не додумывай факты.
Если в контексте нет ответа на вопрос или данных недостаточно, верни ровно эту фразу без пояснений:
{normalized_refusal}

Отвечай кратко и по делу, на русском языке. Если ответ основан на конкретных разделах, упомяни doc_id или название раздела.

Контекст:
{context}

Вопрос:
{normalized_question}

Ответ:"""


def generate_answer(
    question: str,
    chunks: Sequence[RetrievedChunk],
    llm_client: OpenAIChatClient,
    generation_settings: GenerationSettings,
) -> tuple[str, LlmResponse, str]:
    """Генерирует RAG-ответ по вопросу и найденным чанкам.

    Аргументы:
        question: Пользовательский вопрос.
        chunks: Retrieval-чанки для контекста.
        llm_client: OpenAI-compatible клиент генератора.
        generation_settings: Настройки политики генерации.

    Возвращает:
        Кортеж ``(answer, llm_response, prompt)``.
    """

    prompt = build_rag_prompt(question, chunks, generation_settings.refusal_phrase)
    llm_response = llm_client.complete(prompt)
    answer = llm_response.content.strip()
    return answer, llm_response, prompt


def run_rag_pipeline(
    question: str,
    *,
    collection: Any,
    embedding_model: Any,
    bm25_index: BM25Index,
    settings: Settings,
    llm_client: OpenAIChatClient | None = None,
    top_k: int | None = None,
    where: dict[str, Any] | None = None,
    retriever: Callable[..., list[RetrievedChunk]] = hybrid_retrieve,
) -> RagResponse:
    """Запускает полный RAG-пайплайн: retrieval → context → generation.

    Аргументы:
        question: Пользовательский вопрос.
        collection: Chroma collection с индексированными чанками.
        embedding_model: Embedding wrapper с методом ``embed_query``.
        bm25_index: BM25-индекс по тем же чанкам.
        settings: Полные настройки проекта.
        llm_client: Готовый LLM-клиент. Если не передан, создаётся по
            ``settings.llm``.
        top_k: Необязательное переопределение числа чанков в контексте.
        where: Необязательный metadata-фильтр retrieval.
        retriever: Retrieval-функция с интерфейсом ``hybrid_retrieve``.

    Возвращает:
        ``RagResponse`` с ответом, источниками и диагностикой.
    """

    normalized_question = _require_non_empty_string(question, "question")
    requested_top_k = _resolve_top_k(settings.retrieval, top_k)
    chunks = retriever(
        collection,
        embedding_model,
        bm25_index,
        normalized_question,
        settings.retrieval,
        top_k=requested_top_k,
        where=where,
    )
    active_client = llm_client or create_chat_client(settings.llm)
    answer, llm_response, prompt = generate_answer(normalized_question, chunks, active_client, settings.generation)
    return RagResponse(
        question=normalized_question,
        answer=answer,
        retrieved_chunks=list(chunks),
        sources=build_sources(chunks),
        prompt=prompt,
        diagnostics={
            "retrieval_method": "hybrid_rrf" if settings.retrieval.rrf_enabled and settings.retrieval.bm25_enabled else "configured_retriever",
            "top_k": requested_top_k,
            "retrieved_chunks": len(chunks),
            "retrieved_doc_ids": [chunk.doc_id for chunk in chunks],
            "llm_model": llm_response.model_name,
            "llm_usage": llm_response.usage,
            "thinking_disabled": True,
        },
    )


def build_sources(chunks: Sequence[RetrievedChunk]) -> list[dict[str, Any]]:
    """Формирует компактный список источников ответа.

    Аргументы:
        chunks: Retrieval-чанки, вошедшие в контекст.

    Возвращает:
        Список словарей с ``rank``, ``doc_id``, ``title``, ``category`` и score.
    """

    sources: list[dict[str, Any]] = []
    for chunk in chunks:
        sources.append(
            {
                "rank": chunk.rank,
                "doc_id": chunk.doc_id,
                "title": chunk.metadata.get("title", ""),
                "category": chunk.metadata.get("category", ""),
                "chunk_id": chunk.chunk_id,
                "score": round(chunk.score, 6) if chunk.score is not None else None,
                "distance": round(chunk.distance, 6) if chunk.distance is not None else None,
                "method": chunk.retrieval_method,
            }
        )
    return sources


def extract_message_content(response: Any) -> str:
    """Извлекает финальный ответ из ``choices[0].message.content``.

    Аргументы:
        response: Ответ OpenAI-compatible chat-completion API.

    Возвращает:
        Непустой текст ``message.content``.

    Исключения:
        RuntimeError: Если ``message.content`` отсутствует или пустой. Наличие
            текста только в ``reasoning_content`` намеренно считается ошибкой.
    """

    choices = _get_attr_or_key(response, "choices", None)
    if not choices:
        raise RuntimeError("LLM не вернула choices; невозможно получить choices[0].message.content.")
    first_choice = choices[0]
    message = _get_attr_or_key(first_choice, "message", None)
    content = _get_attr_or_key(message, "content", None)
    if isinstance(content, list):
        content = "".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    if isinstance(content, str) and content.strip():
        return content.strip()

    reasoning_content = _get_attr_or_key(message, "reasoning_content", None)
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        raise RuntimeError(
            "LLM вернула текст только в message.reasoning_content, а message.content пуст. "
            "Проверьте передачу chat_template_kwargs.enable_thinking=false."
        )
    raise RuntimeError("LLM вернула пустой choices[0].message.content.")


def validate_llm_settings(settings: LlmSettings) -> None:
    """Проверяет настройки генератора ответов.

    Аргументы:
        settings: Блок ``settings.llm``.

    Исключения:
        ValueError: Если настройки некорректны.
    """

    _require_non_empty_string(settings.base_url, "LLM_BASE_URL")
    _require_non_empty_string(settings.api_key, "LLM_API_KEY")
    _require_non_empty_string(settings.model_name, "LLM_MODEL_NAME")
    if float(settings.temperature) < 0:
        raise ValueError(f"LLM_TEMPERATURE не должен быть отрицательным, получено: {settings.temperature}.")
    if int(settings.timeout_seconds) <= 0:
        raise ValueError(f"LLM_TIMEOUT_SECONDS должен быть положительным, получено: {settings.timeout_seconds}.")
    if int(settings.max_tokens) <= 0:
        raise ValueError(f"LLM_MAX_TOKENS должен быть положительным, получено: {settings.max_tokens}.")


def _resolve_top_k(settings: RetrievalSettings, top_k: int | None) -> int:
    """Возвращает положительное число retrieval-результатов."""

    value = settings.top_k if top_k is None else int(top_k)
    if value <= 0:
        raise ValueError(f"top_k должен быть положительным числом, получено: {value}.")
    return value


def _require_non_empty_string(value: str, name: str) -> str:
    """Проверяет и нормализует непустую строку."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} должен быть непустой строкой.")
    return value.strip()


def _trim_text(text: str, limit: int) -> str:
    """Обрезает текст до лимита символов с многоточием."""

    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "…"


def _get_attr_or_key(value: Any, name: str, default: Any) -> Any:
    """Читает поле из объекта или словаря."""

    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _object_to_dict(value: Any) -> dict[str, Any]:
    """Преобразует объект ответа API в словарь для диагностики."""

    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "dict"):
        return dict(value.dict())
    return {"value": str(value)}