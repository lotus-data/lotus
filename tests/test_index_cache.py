import shutil
from pathlib import Path

import pandas as pd
import pytest

import lotus
from lotus.models import SentenceTransformersRM
from lotus.sem_ops.sem_join import run_sem_sim_join
from lotus.utils import get_index_cache
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

    def test_required_index_dir_parameter(self, sample_df):
        """
        Test that index_dir parameter is required to prevent column name collisions
        """
        df = sample_df.copy()

        # explicit index_dir provided
        df_indexed = df.sem_index("category", "explicit_index_dir")

        # Should use the explicit directory
        cache_dir = df_indexed.attrs["index_dirs"]["category"]
        assert cache_dir == "explicit_index_dir"
        assert Path(cache_dir).exists()

        # cleanup
        if Path("explicit_index_dir").exists():
            shutil.rmtree("explicit_index_dir")

    def test_user_specified_index_directories(self, sample_df, sample_df_2):
        """
        Test that user-specified index directories work correctly
        """
        df1 = sample_df.copy()
        df2 = sample_df_2.copy()

        df_indexed = df1.sem_index("category", "custom_index_name")
        df_indexed_smart = df2.sem_index("category", "custom_index_name_2")

        dir_one = df_indexed.attrs["index_dirs"]["category"]
        dir_two = df_indexed_smart.attrs["index_dirs"]["category"]

        assert dir_one == "custom_index_name"
        assert dir_two == "custom_index_name_2"
        assert Path(dir_one).exists()
        assert Path(dir_two).exists()
        assert dir_one != dir_two

        # delete added paths
        if Path("custom_index_name").exists():
            shutil.rmtree("custom_index_name")
        if Path("custom_index_name_2").exists():
            shutil.rmtree("custom_index_name_2")

    def test_cache_directory_location(self, sample_df):
        """Test that cache is created in specified directory"""
        df = sample_df.copy()
        df_indexed = df.sem_index("category", "test_cache_dir")

        cache_dir = df_indexed.attrs["index_dirs"]["category"]
        assert cache_dir == "test_cache_dir"
        assert Path(cache_dir).exists()

        # cleanup
        if Path("test_cache_dir").exists():
            shutil.rmtree("test_cache_dir")

    def test_cache_files_exist(self, sample_df):
        """
        Test that cache creates both 'index' and 'vecs' files (FAISS creates this)
        """
        df = sample_df.copy()
        df_indexed = df.sem_index("category", "test_files_dir")
        cache_dir = df_indexed.attrs["index_dirs"]["category"]

        assert Path(cache_dir).exists()
        assert (Path(cache_dir) / "index").exists()
        assert (Path(cache_dir) / "vecs").exists()
        assert (Path(cache_dir) / "metadata.json").exists()

        # cleanup
        if Path("test_files_dir").exists():
            shutil.rmtree("test_files_dir")

    def test_data_consistency(self, sample_df, sample_df_2):
        """
        Test that calling sem_index with same data reuses the cache, while
        different data creates new cache
        """
        # same data should reuse cache
        df1 = sample_df.copy()
        df2 = sample_df.copy()
        df3 = sample_df_2.copy()

        # get the dirs
        first_cache_dir = df1.sem_index("category", "test_consistency_1").attrs["index_dirs"]["category"]
        second_cache_dir = df2.sem_index("category", "test_consistency_1").attrs["index_dirs"]["category"]
        third_cache_dir = df3.sem_index("category", "test_consistency_2").attrs["index_dirs"]["category"]

        # same data reuses cache, different data creates new
        assert first_cache_dir == second_cache_dir
        assert first_cache_dir != third_cache_dir
        assert second_cache_dir != third_cache_dir

        # cleanup
        if Path("test_consistency_1").exists():
            shutil.rmtree("test_consistency_1")
        if Path("test_consistency_2").exists():
            shutil.rmtree("test_consistency_2")

    def test_override_flag(self, sample_df):
        """
        Test that override flag forces recreation of index
        """
        df = sample_df.copy()

        # Create initial index
        df.sem_index("category", "test_override")

        # Verify index exists
        assert Path("test_override").exists()
        assert (Path("test_override") / "index").exists()
        assert (Path("test_override") / "metadata.json").exists()

        # Create index with override=True - should not raise error
        # and should recreate the index
        df.sem_index("category", "test_override", override=True)

        # Verify index still exists and is valid
        assert Path("test_override").exists()
        assert (Path("test_override") / "index").exists()
        assert (Path("test_override") / "metadata.json").exists()

        # cleanup
        if Path("test_override").exists():
            shutil.rmtree("test_override")

    def test_data_inconsistency_raises_error(self, sample_df, sample_df_2):
        """
        Test that inconsistent data raises error without override
        """
        df1 = sample_df.copy()
        df2 = sample_df_2.copy()  # Different data

        # Create index with first dataset
        df1.sem_index("category", "test_inconsistent")

        # Try to use same directory with different data - should raise error
        with pytest.raises(ValueError, match="data is inconsistent"):
            df2.sem_index("category", "test_inconsistent")

        # cleanup
        if Path("test_inconsistent").exists():
            shutil.rmtree("test_inconsistent")

    def test_sem_join_no_duplicate_index_creation(self, sample_df):
        """
        Test that sem_join_cascade doesn't create duplicate indices
        """
        df1 = sample_df.copy()
        df2 = sample_df.copy()

        # Track number of index creations by checking logs or index files
        # First call
        run_sem_sim_join(df1["category"], df2["category"], "cat1", "cat2")

        # Get cache directory
        rm = lotus.settings.rm
        model_name = getattr(rm, "model", None)
        cache_dir = get_index_cache("cat2", df2["category"].tolist(), model_name)
        initial_mtime = Path(cache_dir).stat().st_mtime

        # Second call with same l2 data - should hit cache
        run_sem_sim_join(df1["category"], df2["category"], "cat1_mapped", "cat2")

        # Verify cache directory wasn't recreated (same mtime)
        assert Path(cache_dir).stat().st_mtime == initial_mtime

        # cleanup
        if Path(cache_dir).exists():
            shutil.rmtree(cache_dir)
