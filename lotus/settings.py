import lotus.models


class Settings:
    # Models
    lm: lotus.models.LM | None = None
    rm: lotus.models.RM | None = None
    helper_lm: lotus.models.LM | None = None
    reranker: lotus.models.Reranker | None = None

    # Cache settings
    enable_cache: bool = False

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid setting: {key}")
            setattr(self, key, value)


settings = Settings()
