import asyncio
import threading

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
        assert settings.enable_cache is False
        assert settings.serialization_format == SerializationFormat.DEFAULT

    def test_configure_method(self, settings):
        settings.configure(enable_cache=True)
        assert settings.enable_cache is True

    def test_invalid_setting(self, settings):
        with pytest.raises(ValueError, match="Invalid setting: invalid_setting"):
            settings.configure(invalid_setting=True)


class TestSettingsContext:
    @pytest.fixture
    def settings(self):
        return Settings()

    def test_context_restores_on_exit(self, settings):
        settings.configure(enable_cache=True)
        with settings.context(enable_cache=False):
            assert settings.enable_cache is False
        assert settings.enable_cache is True

    def test_context_restores_class_default_on_exit(self, settings):
        # enable_cache starts at class default (False, not set as instance attr)
        assert "enable_cache" not in vars(settings)
        with settings.context(enable_cache=True):
            assert settings.enable_cache is True
        # ContextVar is reset; class default takes over without touching instance dict
        assert "enable_cache" not in vars(settings)
        assert settings.enable_cache is False

    def test_context_restores_on_exception(self, settings):
        settings.configure(enable_cache=True)
        with pytest.raises(RuntimeError):
            with settings.context(enable_cache=False):
                assert settings.enable_cache is False
                raise RuntimeError("boom")
        assert settings.enable_cache is True

    def test_context_yields_settings(self, settings):
        with settings.context(enable_cache=True) as s:
            assert s is settings
            assert s.enable_cache is True

    def test_context_multiple_overrides(self, settings):
        settings.configure(enable_cache=True, parallel_groupby_max_threads=4)
        with settings.context(enable_cache=False, parallel_groupby_max_threads=16):
            assert settings.enable_cache is False
            assert settings.parallel_groupby_max_threads == 16
        assert settings.enable_cache is True
        assert settings.parallel_groupby_max_threads == 4

    def test_nested_contexts(self, settings):
        settings.configure(enable_cache=False)
        with settings.context(enable_cache=True):
            assert settings.enable_cache is True
            with settings.context(enable_cache=False):
                assert settings.enable_cache is False
            assert settings.enable_cache is True
        assert settings.enable_cache is False

    def test_context_invalid_setting_raises(self, settings):
        settings.configure(enable_cache=True)
        with pytest.raises(ValueError, match="Invalid setting: bad_key"):
            with settings.context(bad_key=True):
                pass  # pragma: no cover
        # Settings must be unchanged after the failed context entry
        assert settings.enable_cache is True

    def test_context_serialization_format(self, settings):
        settings.configure(serialization_format=SerializationFormat.JSON)
        with settings.context(serialization_format=SerializationFormat.XML):
            assert settings.serialization_format == SerializationFormat.XML
        assert settings.serialization_format == SerializationFormat.JSON


class TestSettingsContextConcurrency:
    @pytest.fixture
    def settings(self):
        return Settings()

    def test_thread_isolation(self, settings):
        """Two threads entering context() simultaneously see only their own overrides."""
        results: dict[int, bool] = {}
        barrier = threading.Barrier(2)

        def run(thread_id: int, value: bool) -> None:
            with settings.context(enable_cache=value):
                barrier.wait()  # both threads inside context at the same time
                results[thread_id] = settings.enable_cache

        t1 = threading.Thread(target=run, args=(1, True))
        t2 = threading.Thread(target=run, args=(2, False))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results[1] is True
        assert results[2] is False

    def test_thread_baseline_unaffected(self, settings):
        """Global baseline is unchanged after threads exit their contexts."""
        settings.configure(enable_cache=False)
        barrier = threading.Barrier(2)

        def run(value: bool) -> None:
            with settings.context(enable_cache=value):
                barrier.wait()

        threads = [threading.Thread(target=run, args=(v,)) for v in (True, False)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert settings.enable_cache is False

    def test_asyncio_task_isolation(self, settings):
        """Two asyncio tasks entering context() see only their own overrides."""

        async def run(value: bool) -> bool:
            with settings.context(enable_cache=value):
                await asyncio.sleep(0)  # yield so both tasks overlap
                return settings.enable_cache

        async def main() -> tuple[bool, bool]:
            return await asyncio.gather(run(True), run(False))

        r_true, r_false = asyncio.run(main())
        assert r_true is True
        assert r_false is False
