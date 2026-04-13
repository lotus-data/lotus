from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Generator

import lotus.models
import lotus.vector_store
from lotus.types import SerializationFormat

# context() is safe for concurrent use across threads and asyncio tasks.
# Direct mutation via configure() or attribute assignment is not thread-safe.

_settings_context: ContextVar[dict[str, Any] | None] = ContextVar("_settings_context", default=None)


class Settings:
    # Models
    lm: lotus.models.LM | None = None
    rm: lotus.models.RM | None = None  # supposed to only generate embeddings
    helper_lm: lotus.models.LM | None = None
    reranker: lotus.models.Reranker | None = None
    vs: lotus.vector_store.VS | None = None

    # Cache settings
    enable_cache: bool = False

    # Serialization setting
    serialization_format: SerializationFormat = SerializationFormat.DEFAULT

    # Parallel groupby settings
    parallel_groupby_max_threads: int = 8

    def __getattribute__(self, name: str) -> Any:
        # For known settings fields, check the per-context overlay first.
        annotations = object.__getattribute__(self, "__class__").__annotations__
        if name in annotations:
            ctx = _settings_context.get()
            if ctx is not None and name in ctx:
                return ctx[name]
        return object.__getattribute__(self, name)

    def configure(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid setting: {key}")
            setattr(self, key, value)

    @contextmanager
    def context(self, **kwargs: Any) -> Generator["Settings", None, None]:
        """Temporarily override settings in the current thread or asyncio task.

        Each thread and asyncio task sees only its own overrides — concurrent
        callers cannot interfere with each other. Supports nesting and
        guarantees restoration even if an exception is raised.

        Example::

            with lotus.settings.context(enable_cache=False, lm=eval_lm):
                result = df.sem_filter("...")
        """
        # Validate all keys before making any changes.
        for key in kwargs:
            if not hasattr(self, key):
                raise ValueError(f"Invalid setting: {key}")

        current = _settings_context.get() or {}
        token = _settings_context.set({**current, **kwargs})
        try:
            yield self
        finally:
            _settings_context.reset(token)

    def __str__(self) -> str:
        return str(vars(self))


settings = Settings()
