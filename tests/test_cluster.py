import pandas as pd
import pytest

from tests.base_test import BaseTest


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Course Name": [
                "Probability and Random Processes",
                "Statistics and Data Analysis",
                "Cooking Basics",
                "Advanced Culinary Arts",
                "Digital Circuit Design",
                "Computer Architecture",
            ]
        }
    )


class TestClusterBy(BaseTest):
    def test_basic_clustering(self, sample_df):
        """Test basic clustering functionality with 2 clusters"""
        result = sample_df.sem_cluster_by("Course Name", 2)
        assert "cluster_id" in result.columns
        assert len(result["cluster_id"].unique()) == 2
        assert len(result) == len(sample_df)

        # Get the two clusters
        cluster_0_courses = set(result[result["cluster_id"] == 0]["Course Name"])
        cluster_1_courses = set(result[result["cluster_id"] == 1]["Course Name"])

        # Define the expected course groupings
        tech_courses = {
            "Probability and Random Processes",
            "Statistics and Data Analysis",
            "Digital Circuit Design",
            "Computer Architecture",
        }
        culinary_courses = {"Cooking Basics", "Advanced Culinary Arts"}

        # Check that one cluster contains tech courses and the other contains culinary courses
        assert (cluster_0_courses == tech_courses and cluster_1_courses == culinary_courses) or (
            cluster_1_courses == tech_courses and cluster_0_courses == culinary_courses
        ), "Clusters don't match expected course groupings"

    def test_clustering_with_more_clusters(self, sample_df):
        """Test clustering with more clusters than necessary"""
        result = sample_df.sem_cluster_by("Course Name", 3)
        assert len(result["cluster_id"].unique()) == 3
        assert len(result) == len(sample_df)

    def test_clustering_with_single_cluster(self, sample_df):
        """Test clustering with single cluster"""
        result = sample_df.sem_cluster_by("Course Name", 1)
        assert len(result["cluster_id"].unique()) == 1
        assert result["cluster_id"].iloc[0] == 0

    def test_clustering_with_invalid_column(self, sample_df):
        """Test clustering with non-existent column"""
        with pytest.raises(ValueError, match="Column .* not found in DataFrame"):
            sample_df.sem_cluster_by("NonExistentColumn", 2)

    def test_clustering_with_empty_dataframe(self):
        """Test clustering on empty dataframe"""
        empty_df = pd.DataFrame(columns=["Course Name"])
        result = empty_df.sem_cluster_by("Course Name", 2)
        assert len(result) == 0
        assert "cluster_id" in result.columns

    def test_clustering_similar_items(self, sample_df):
        """Test that similar items are clustered together"""
        result = sample_df.sem_cluster_by("Course Name", 3)

        # Get cluster IDs for similar courses
        stats_cluster = result[result["Course Name"].str.contains("Statistics")]["cluster_id"].iloc[0]
        prob_cluster = result[result["Course Name"].str.contains("Probability")]["cluster_id"].iloc[0]

        # Similar courses should be in the same cluster
        assert stats_cluster == prob_cluster

        cooking_cluster = result[result["Course Name"].str.contains("Cooking")]["cluster_id"].iloc[0]
        culinary_cluster = result[result["Course Name"].str.contains("Culinary")]["cluster_id"].iloc[0]

        assert cooking_cluster == culinary_cluster

    def test_clustering_with_verbose(self, sample_df):
        """Test clustering with verbose output"""
        result = sample_df.sem_cluster_by("Course Name", 2, verbose=True)
        assert "cluster_id" in result.columns
        assert len(result["cluster_id"].unique()) == 2

    def test_clustering_with_iterations(self, sample_df):
        """Test clustering with different iteration counts"""
        result1 = sample_df.sem_cluster_by("Course Name", 2, niter=5)
        result2 = sample_df.sem_cluster_by("Course Name", 2, niter=20)

        # Both should produce valid clusterings
        assert len(result1["cluster_id"].unique()) == 2
        assert len(result2["cluster_id"].unique()) == 2
