import lotus.models
import lotus.vector_store
from lotus.types import SerializationFormat

# NOTE: Settings class is not thread-safe


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

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid setting: {key}")
            setattr(self, key, value)

    def __str__(self):
        return str(vars(self))


settings = Settings()
