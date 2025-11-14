from lotus.models import LMWithTools
from lotus.tests.base_test import BaseTest


class TestLMWithTools(BaseTest):
    def test_default_lm_with_tools(self, settings):
        """Test that the default LM with tools is None"""
        assert settings.lm_with_tools is None

    def test_lm_with_tools_initialization(self):
        lm = LMWithTools()
        assert isinstance(lm, LMWithTools)

    def test_configure_lm_with_tools(self, settings, lm_with_tools):
        """Test configuring the LM with tools in settings"""
        settings.configure(lm_with_tools=lm_with_tools)
        assert settings.lm_with_tools is lm_with_tools
