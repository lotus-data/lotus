import pytest

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
        assert settings.enable_message_cache is False
        assert settings.serialization_format == SerializationFormat.DEFAULT

    def test_configure_method(self, settings):
        settings.configure(enable_message_cache=True)
        assert settings.enable_message_cache is True

    def test_invalid_setting(self, settings):
        with pytest.raises(ValueError, match="Invalid setting: invalid_setting"):
            settings.configure(invalid_setting=True)
