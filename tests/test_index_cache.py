import shutil
from pathlib import Path

import pandas as pd
import pytest

import lotus
from lotus.models import SentenceTransformersRM
from lotus.vector_store import FaissVS
from tests.base_test import BaseTest

# Set logger level to DEBUG
lotus.logger.setLevel("DEBUG")


@pytest.fixture(scope="module")
def vs():
    return FaissVS()


@pytest.fixture(scope="module")
def rm():
    return SentenceTransformersRM(model="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
def sample_df():
    """
    Sample DataFrame for testing
    """
    return pd.DataFrame({"category": ["Sports", "Food", "Video Games", "STEM"]})


@pytest.fixture
def sample_df_2():
    """
    Second different DataFrame for additional testing
    """
    return pd.DataFrame({"category": ["Sports", "STEM"]})


class TestIndexCache(BaseTest):
    """Test suite for index caching functionality"""

    @pytest.fixture(autouse=True)
    def setup_vs(self, rm, vs):
        lotus.settings.configure(rm=rm, vs=vs)

    def test_backwards_compatibility(self, sample_df, sample_df_2):
        """
        Test backward compatibility where the index directory is passed
        """
        df1 = sample_df.copy()
        df2 = sample_df_2.copy()

        df_indexed = df1.sem_index("category", index_dir="custom_index_name")
        df_indexed_smart = df2.sem_index("category")

        dir_one = df_indexed.attrs["index_dirs"]["category"]
        dir_two = df_indexed_smart.attrs["index_dirs"]["category"]

        assert "/.cache/lotus/indices/" not in dir_one
        assert dir_one == "custom_index_name"
        assert Path(dir_one).exists()
        assert dir_one != dir_two

        # delete added path
        if Path("custom_index_name").exists():
            shutil.rmtree("custom_index_name")

    def test_cache_directory_location(self, sample_df):
        """Test that cache is created in ~/.cache/lotus/indices/"""
        df = sample_df.copy()
        df_indexed = df.sem_index("category")

        cache_dir = df_indexed.attrs["index_dirs"]["category"]
        assert "/.cache/lotus/indices/" in cache_dir

    def test_cache_files_exist(self, sample_df):
        """
        Test that cache creates both 'index' and 'vecs' files (FAISS creates this)
        """
        df = sample_df.copy()
        df_indexed = df.sem_index("category")
        cache_dir = df_indexed.attrs["index_dirs"]["category"]

        assert Path(cache_dir).exists()
        assert (Path(cache_dir) / "index").exists()
        assert (Path(cache_dir) / "vecs").exists()

    def test_dir_name(self, sample_df, sample_df_2):
        """
        Test that calling sem_index twice with same data reuses the cache, while
        different data uses different caches
        """
        # same data should give same dir
        df1 = sample_df.copy()
        df2 = sample_df.copy()
        df3 = sample_df_2.copy()

        # get the dir
        first_cache_dir = df1.sem_index("category").attrs["index_dirs"]["category"]
        second_cache_dir = df2.sem_index("category").attrs["index_dirs"]["category"]
        third_cache_dir = df3.sem_index("category").attrs["index_dirs"]["category"]

        # same data has same directory, different data does not
        assert first_cache_dir == second_cache_dir
        assert first_cache_dir != third_cache_dir
        assert second_cache_dir != third_cache_dir
