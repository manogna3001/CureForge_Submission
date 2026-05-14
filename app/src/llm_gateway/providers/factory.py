import threading
from time import perf_counter
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.src.utils.metrics import get_metrics
from app.src.utils.settings import get_settings


settings = get_settings()


class _LLMMetricsCallback(BaseCallbackHandler):
    def __init__(self, model_name: str):
        super().__init__()
        self._model_name = model_name
        self._timers: dict[str, float] = {}
        self._timer_lock = threading.Lock()

    def on_chat_model_start(
        self, serialized: dict, messages: list, *, run_id: UUID, **kwargs: Any
    ) -> None:
        get_metrics().increment("llm_calls_total")
        with self._timer_lock:
            self._timers[str(run_id)] = perf_counter()

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        with self._timer_lock:
            start = self._timers.pop(str(run_id), None)
        if start is not None:
            get_metrics().histogram(
                "llm_duration_seconds", value=perf_counter() - start
            )
        try:
            if hasattr(response, "llm_output") and isinstance(response.llm_output, dict):
                usage = response.llm_output.get("token_usage", {})
                total = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0
                if total:
                    get_metrics().gauge(
                        "llm_tokens_used",
                        labels={"model": self._model_name},
                        value=float(total),
                    )
        except Exception:
            pass

    def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        with self._timer_lock:
            self._timers.pop(str(run_id), None)
        get_metrics().increment("llm_errors_total")


def get_provider(id: str = None, model_name: str = None, temperature: float = 1):
    """Returns a provider instance based on the specified configuration."""
    resolved_model_name = model_name or settings.model_name

    return ChatOpenAI(
        model=resolved_model_name,
        temperature=temperature,
        api_key=settings.litellm_api_key,
        base_url=settings.litellm_base_url,
        default_headers={"X-Agent-ID": id} if id else None,
        max_retries=3,
        callbacks=[_LLMMetricsCallback(resolved_model_name)],
    )


def get_embedding_provider(model_name: str) -> OpenAIEmbeddings:
    """Returns an OpenAIEmbeddings instance routed through the litellm proxy."""
    return OpenAIEmbeddings(
        model=model_name,
        api_key=settings.litellm_api_key,
        base_url=settings.litellm_base_url,
        max_retries=3,
    )
