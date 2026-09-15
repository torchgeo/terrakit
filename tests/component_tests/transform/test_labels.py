# © Copyright IBM Corporation 2025-2026
# SPDX-License-Identifier: Apache-2.0


import geopandas as gpd
import os
import pandas as pd
import pytest

from shapely.geometry import box
from pathlib import Path

from terrakit.transform.labels import LabelsCls
from terrakit.transform.labels import process_labels
from tests.component_tests.transform.conftest import (
    DATASET_NAME,
    DEFAULT_DATASET_NAME,
    DEFAULT_WORKING_DIR,
    WORKING_DIR,
    LABELS_FOLDER,
    LABELS_FOLDER_RASTER,
    LABELS_FOLDER_CLASSES,
)


class TestLabels_WorkingDir:
    def test_process_labels__working_dir_default(
        self, process_labels_clean_up_default_working_dir
    ):
        """Test shp created as expected when using the default working directory"""
        # Create the default working dir before starting
        Path(DEFAULT_WORKING_DIR).mkdir(parents=True, exist_ok=True)

        process_labels(labels_folder=LABELS_FOLDER)

        # Validate correct output from process_labels
        assert f"{DEFAULT_DATASET_NAME}_all_bboxes.shp" in os.listdir(
            DEFAULT_WORKING_DIR
        )
        assert f"{DEFAULT_DATASET_NAME}_labels.shp" in os.listdir(DEFAULT_WORKING_DIR)

    def test_process_labels__working_dir_created_if_does_not_exist(
        self, process_labels_clean_up_default_working_dir
    ):
        """Test working directory is created if it does not already exist"""
        # Ensure dir does not exist before starting
        assert os.path.exists(DEFAULT_WORKING_DIR) is False

        process_labels(
            labels_folder=LABELS_FOLDER,
        )

        # Validate correct output from process_labels
        assert f"{DEFAULT_DATASET_NAME}_all_bboxes.shp" in os.listdir(
            DEFAULT_WORKING_DIR
        )
        assert f"{DEFAULT_DATASET_NAME}_labels.shp" in os.listdir(DEFAULT_WORKING_DIR)

    def test_process_labels__set_working_dir(
        self,
        process_labels_clean_up_working_dir,
    ):
        """Test working directory can be set to some valid path"""
        labels_gdf, grouped_boxes_gdf = process_labels(
            working_dir=WORKING_DIR,
            labels_folder=LABELS_FOLDER,
        )

        # Validate correct output from process_labels
        assert isinstance(grouped_boxes_gdf, pd.DataFrame)
        assert "datetime" in list(grouped_boxes_gdf.columns.values)
        assert f"{DEFAULT_DATASET_NAME}_all_bboxes.shp" in os.listdir(WORKING_DIR)
        assert isinstance(labels_gdf, pd.DataFrame)
        assert "datetime" in list(labels_gdf.columns.values)
        assert f"{DEFAULT_DATASET_NAME}_labels.shp" in os.listdir(WORKING_DIR)


class TestLabels_NoOverwrite:
    def test_process_no_overwrite(
        self, caplog, process_labels_clean_up_default_working_dir
    ):
        # Note this test only confirms that a Warning is raised if the .shp file already exists.
        labels_gdf, grouped_boxes_gdf = process_labels(
            labels_folder=LABELS_FOLDER,
        )

        # Confirm labels are processed as expected
        assert isinstance(grouped_boxes_gdf, pd.DataFrame)
        assert "datetime" in list(grouped_boxes_gdf.columns.values)
        assert f"{DEFAULT_DATASET_NAME}_all_bboxes.shp" in os.listdir(
            DEFAULT_WORKING_DIR
        )
        assert isinstance(labels_gdf, pd.DataFrame)
        assert "datetime" in list(labels_gdf.columns.values)
        assert f"{DEFAULT_DATASET_NAME}_labels.shp" in os.listdir(DEFAULT_WORKING_DIR)

        # Run twice to confirm file is not overwritten
        process_labels(labels_folder=LABELS_FOLDER)
        assert (
            caplog.records[-1].levelname == "WARNING"
        )  # Check that the last log message is a warning before the function returns.
        assert (
            "_all_bboxes.shp' already exists and will not be overwritten" in caplog.text
        )
        assert "_labels.shp' already exists and will not be overwritten" in caplog.text


class TestLabels_Active:
    def test_process_labels_inactive(
        self, caplog, process_labels_clean_up_default_working_dir
    ):
        labels_gdf, grouped_boxes_gdf = process_labels(
            active=False, labels_folder=LABELS_FOLDER
        )
        assert type(grouped_boxes_gdf) is pd.DataFrame
        assert len(grouped_boxes_gdf) == 0
        assert type(labels_gdf) is pd.DataFrame
        assert len(labels_gdf) == 0
        assert (
            caplog.records[-1].levelname == "WARNING"
        )  # Check that the last log message is as expected before the function returns.
        assert " Labels are not active." in caplog.text
        assert f"{DEFAULT_DATASET_NAME}_all_bboxes.shp" not in os.listdir(
            DEFAULT_WORKING_DIR
        )


class TestLabels_DatatimeInfo:
    """Test process_labels works as expected with datetime_info set"""

    def test_process_labels__datetime_info__file(
        self,
        process_labels_clean_up_default_working_dir,
    ):
        process_labels(
            labels_folder=LABELS_FOLDER,
            datetime_info="filename",
        )

        assert f"{DEFAULT_DATASET_NAME}_all_bboxes.shp" in os.listdir(
            DEFAULT_WORKING_DIR
        )


class TestLabels_LabelType:
    @pytest.mark.parametrize(
        "label_type, labels_folder",
        (["vector", LABELS_FOLDER], ["raster", LABELS_FOLDER_RASTER]),
    )
    def test_process_labels__label_type(
        self,
        process_labels_clean_up_default_working_dir,
        label_type,
        labels_folder,
        caplog,
    ):
        labels_gdf, grouped_boxes_gdf = process_labels(
            labels_folder=labels_folder,
            label_type=label_type,
        )
        assert isinstance(grouped_boxes_gdf, pd.DataFrame)
        assert len(grouped_boxes_gdf) > 0
        assert isinstance(labels_gdf, pd.DataFrame)
        assert len(labels_gdf) > 0
        assert "2/2 label files were successfully processed." in caplog.text


class TestLabels_DatetimeInfo:
    def test_process_labels__csv(
        self,
        process_labels_setup_csv_datetime,
        process_labels_clean_up_default_working_dir,
        process_labels_clean_up_csv_labels_dir,
        caplog,
    ):
        labels_gdf, grouped_boxes_gdf = process_labels(
            labels_folder="docs/examples/test_wildfire_vector_metadata_csv",
            datetime_info="csv",
        )
        assert isinstance(grouped_boxes_gdf, pd.DataFrame)
        assert len(grouped_boxes_gdf) > 0
        assert isinstance(labels_gdf, pd.DataFrame)
        assert len(labels_gdf) > 0
        assert "2/2 label files were successfully processed." in caplog.text


class TestLabels_Provenance:
    def test_process_labels__provenance(
        self,
        process_labels_clean_up_working_dir,
        caplog,
    ):
        """Test working directory can be set to some valid path"""
        labels_gdf, grouped_boxes_gdf = process_labels(
            dataset_name=DATASET_NAME,
            working_dir=WORKING_DIR,
            labels_folder=LABELS_FOLDER,
        )
        num_files = (
            2 * 5 + 1
        )  # 2 shapefiles collections, each with 5 files, plus 1 data stat provenance file.
        assert len(os.listdir(Path(WORKING_DIR))) == num_files
        assert f"{DATASET_NAME}_metadata.json" in os.listdir(Path(WORKING_DIR))


class TestLabels_Classes:
    def test_process_labels__classes(
        self,
        process_labels_clean_up_working_dir,
        caplog,
    ):
        """Test process_labels works with multi-class label files"""
        labels_gdf, grouped_boxes_gdf = process_labels(
            dataset_name=DATASET_NAME,
            working_dir=WORKING_DIR,
            labels_folder=LABELS_FOLDER_CLASSES,
        )

        num_files = (
            2 * 5 + 1
        )  # 2 shapefiles collections, each with 5 files, plus 1 data stat provenance file.
        assert len(os.listdir(Path(WORKING_DIR))) == num_files
        assert f"{DATASET_NAME}_metadata.json" in os.listdir(Path(WORKING_DIR))

        # Verify multi-class labels were processed
        assert isinstance(grouped_boxes_gdf, pd.DataFrame)
        assert len(grouped_boxes_gdf) > 0
        assert isinstance(labels_gdf, pd.DataFrame)
        assert len(labels_gdf) > 0
        assert "2/2 label files were successfully processed." in caplog.text

        # Verify class 0 and class 1 are present in labels
        assert "labelclass" in labels_gdf.columns
        assert 0 in labels_gdf["labelclass"].values, (
            "Class 0 should be present in labels"
        )
        assert 1 in labels_gdf["labelclass"].values, (
            "Class 1 should be present in labels"
        )

        # Verify both classes are in the grouped boxes
        assert "labelclass" in grouped_boxes_gdf.columns
        assert set(grouped_boxes_gdf["labelclass"].values) == {0, 1}, (
            "Both class 0 and 1 should be in grouped boxes"
        )


class TestOverlappingGeometries:
    """Test that non-overlapping geometries are grouped separately with tile suffixes."""

    def test_non_overlapping_geometries_get_separate_tile_suffixes(self):
        """Test that non-overlapping geometries on the same date get different tile suffixes."""
        # Create test data with two non-overlapping geometries on the same date
        test_data = {
            "datetime": ["2024-01-01", "2024-01-01"],
            "labelclass": ["fire", "fire"],
            "filename": ["file1.shp", "file2.shp"],  # Add filename column
            "geometry": [
                box(0, 0, 1, 1),  # First bbox
                box(5, 5, 6, 6),  # Second bbox (non-overlapping)
            ],
        }
        test_gdf = gpd.GeoDataFrame(test_data, crs="EPSG:4326")

        # Create LabelsCls instance
        labels = LabelsCls(
            dataset_name="test_dataset",
            working_dir="./tmp_test",
            labels_folder="./test_labels",
        )

        # Get grouped bboxes
        result_gdf = labels.get_grouped_bbox_gdf(test_gdf)

        # Verify that we have a tilesuffix column
        assert "tilesuffix" in result_gdf.columns

        # Verify that we have two separate groups (two rows for same date)
        assert len(result_gdf) == 2

        # Verify that the tile suffixes are different
        tile_suffixes = result_gdf["tilesuffix"].unique()
        assert len(tile_suffixes) == 2

        # Verify tile suffix format (e.g. "_tile_0_0_", "_tile_0_1_")
        for suffix in tile_suffixes:
            assert suffix.startswith("_tile_")

    def test_overlapping_geometries_get_same_tile_suffix(self):
        """Test that overlapping geometries on the same date get the same tile suffix."""
        # Create test data with two overlapping geometries on the same date
        test_data = {
            "datetime": ["2024-01-01", "2024-01-01"],
            "labelclass": ["fire", "smoke"],
            "filename": ["file1.shp", "file1.shp"],  # Add filename column (same file)
            "geometry": [
                box(0, 0, 2, 2),  # First bbox
                box(1, 1, 3, 3),  # Second bbox (overlapping)
            ],
        }
        test_gdf = gpd.GeoDataFrame(test_data, crs="EPSG:4326")

        # Create LabelsCls instance
        labels = LabelsCls(
            dataset_name="test_dataset",
            working_dir="./tmp_test",
            labels_folder="./test_labels",
        )

        # Get grouped bboxes
        result_gdf = labels.get_grouped_bbox_gdf(test_gdf)

        # Verify that we have a tile_suffix column
        assert "tilesuffix" in result_gdf.columns

        # Verify that we have two rows (one per class) but same tile suffix
        assert len(result_gdf) == 2

        # Verify that both have the same tile suffix
        tile_suffixes = result_gdf["tilesuffix"].unique()
        assert len(tile_suffixes) == 1

    def test_mixed_overlapping_and_non_overlapping_geometries(self):
        """Test mixed scenario with both overlapping and non-overlapping geometries."""
        # Create test data:
        # - Geometries 0 and 1 overlap (group 1)
        # - Geometry 2 is separate (group 2)
        test_data = {
            "datetime": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "labelclass": ["fire", "smoke", "fire"],
            "filename": ["file1.shp", "file1.shp", "file2.shp"],  # Add filename column
            "geometry": [
                box(0, 0, 2, 2),  # Overlaps with geometry 1
                box(1, 1, 3, 3),  # Overlaps with geometry 0
                box(10, 10, 12, 12),  # Separate
            ],
        }
        test_gdf = gpd.GeoDataFrame(test_data, crs="EPSG:4326")

        # Create LabelsCls instance
        labels = LabelsCls(
            dataset_name="test_dataset",
            working_dir="./tmp_test",
            labels_folder="./test_labels",
        )

        # Get grouped bboxes
        result_gdf = labels.get_grouped_bbox_gdf(test_gdf)

        # Verify that we have a tile_suffix column
        assert "tilesuffix" in result_gdf.columns

        # Verify that we have 3 rows (one per class)
        assert len(result_gdf) == 3

        # Verify that we have exactly 2 different tile suffixes
        tile_suffixes = result_gdf["tilesuffix"].unique()
        assert len(tile_suffixes) == 2

    def test_different_dates_get_independent_tile_suffixes(self):
        """Test that different dates are processed independently."""
        # Create test data with non-overlapping geometries on different dates
        test_data = {
            "datetime": ["2024-01-01", "2024-01-02"],
            "labelclass": ["fire", "fire"],
            "filename": ["file1.shp", "file2.shp"],  # Add filename column
            "geometry": [
                box(0, 0, 1, 1),  # Date 1
                box(
                    5, 5, 6, 6
                ),  # Date 2 (same position as would be non-overlapping on date 1)
            ],
        }
        test_gdf = gpd.GeoDataFrame(test_data, crs="EPSG:4326")

        # Create LabelsCls instance
        labels = LabelsCls(
            dataset_name="test_dataset",
            working_dir="./tmp_test",
            labels_folder="./test_labels",
        )

        # Get grouped bboxes
        result_gdf = labels.get_grouped_bbox_gdf(test_gdf)

        # Verify that we have a tile_suffix column
        assert "tilesuffix" in result_gdf.columns

        # Verify that we have 2 rows (one per date)
        assert len(result_gdf) == 2

        # Each date should have its own tile suffix
        # Format: _tile_YYYYMMDD_M_N_ where M is date index (0, 1, 2...) and N is subgroup index (1, 2, 3...)
        date1_row = result_gdf[result_gdf["datetime"] == "2024-01-01"].iloc[0]
        date2_row = result_gdf[result_gdf["datetime"] == "2024-01-02"].iloc[0]

        assert date1_row["tilesuffix"] == "_tile_0_1"
        assert date2_row["tilesuffix"] == "_tile_1_1"
