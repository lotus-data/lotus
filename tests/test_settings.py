import pytest

from lotus.models import LM
from lotus.settings import SerializationFormat, Settings


class TestSettings:
    @pytest.fixture
    def settings(self):
        return Settings()

    def test_initial_values(self, settings):
        assert settings.lm is None
        assert settings.rm is None
        assert settings.helper_lm is None
        assert settings.reranker is None
        assert settings.enable_cache is False
        assert settings.serialization_format == SerializationFormat.DEFAULT
        assert settings.enable_multithreading is False

    def test_configure_method(self, settings):
        settings.configure(enable_multithreading=True)
        assert settings.enable_multithreading is True

    def test_invalid_setting(self, settings):
        with pytest.raises(ValueError, match="Invalid setting: invalid_setting"):
            settings.configure(invalid_setting=True)

    def test_clone_method(self, settings):
        other_settings = Settings()
        lm = LM(model="test-model")
        other_settings.lm = lm
        other_settings.enable_cache = True

        settings.clone(other_settings)

        assert settings.lm == lm
        assert settings.enable_cache is True
