import pytest

from lotus.models import LM, LMWithoutTools, LMWithTools
from lotus.settings import Settings
from tests.base_test import BaseTest


class TestLMWithTools(BaseTest):
    @pytest.fixture
    def settings(self):
        return Settings()

    def test_lm_without_tools_initialization(self):
        lm = LM()
        assert isinstance(lm, LMWithoutTools)

    def test_configure_lm_with_tools(self, settings):
        """Test configuring the LM with tools in settings"""
        settings.configure(lm=LM(model="gpt-4o-mini", with_tools=True))
        assert settings.lm is not None
        assert isinstance(settings.lm, LMWithTools)

        settings.configure(lm=None)
        assert settings.lm is None

        settings.configure(lm=LM(with_tools=True))
        assert settings.lm is not None
        assert isinstance(settings.lm, LMWithTools)

        settings.configure(lm=None)
        assert settings.lm is None

        settings.configure(lm=LM("gpt-4o-mini", max_batch_size=64, with_tools=True))
        assert settings.lm is not None
        assert isinstance(settings.lm, LMWithTools)

        settings.configure(lm=None)
        assert settings.lm is None
