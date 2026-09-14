# © Copyright IBM Corporation 2025-2026
# SPDX-License-Identifier: Apache-2.0


import os
import pandas as pd
import pytest

from pathlib import Path

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
