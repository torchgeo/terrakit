# © Copyright IBM Corporation 2025-2026
# SPDX-License-Identifier: Apache-2.0


import csv
import os
import pytest
import shutil

from glob import glob
from pathlib import Path

from terrakit.general_utils.labels_downloader import (
    rapid_mapping_geojson_downloader,
    rapid_mapping_class_split,
    hugging_face_file_downloader,
    EXAMPLE_LABEL_FILES,
    EXAMPLE_RASTER_LABEL_FILES,
    EXAMPLE_CLASS_LABEL_FILES,
)

DATASET_NAME = "terrakit_testing"
DEFAULT_DATASET_NAME = "terrakit_curated_dataset"
WORKING_DIR: str = "./tests/resources/component_test_data/transform/tmp"
DEFAULT_WORKING_DIR: str = "./tmp"
LABELS_FOLDER: str = "./docs/examples/test_wildfire_vector"
LABELS_FOLDER_RASTER: str = "./docs/examples/test_burn_scar_raster"
LABELS_FOLDER_CSV_DATETIME: str = "./docs/examples/test_wildfire_vector_metadata_csv"
LABELS_FOLDER_CLASSES: str = "./docs/examples/test_wildfire_classes_vector"
EMPTY_LABELS_FOLDER: str = "./tests/tmp/labels"


@pytest.fixture(scope="class", autouse=True)
def download_example_labels():
    if (
        Path(LABELS_FOLDER).is_dir() is False
        or set(EXAMPLE_LABEL_FILES).issubset(
            [Path(f).name for f in glob(f"{LABELS_FOLDER}/*.json")]
        )
        is False
    ):
        rapid_mapping_geojson_downloader(
            event_id="748",
            aoi="01",
            monitoring_number="05",
            version="v1",
            dest=LABELS_FOLDER,
        )
        rapid_mapping_geojson_downloader(
            event_id="801",
            aoi="01",
            monitoring_number="02",
            version="v1",
            dest=LABELS_FOLDER,
        )

    if (
        Path(LABELS_FOLDER_RASTER).is_dir() is False
        or set(EXAMPLE_RASTER_LABEL_FILES).issubset(
            [Path(f).name for f in glob(f"{LABELS_FOLDER_RASTER}/*.tif")]
        )
        is False
    ):
        for filename in EXAMPLE_RASTER_LABEL_FILES:
            hugging_face_file_downloader(
                repo_id="ibm-nasa-geospatial/hls_burn_scars",
                filename=filename,
                revision="e48662b31288f1d5f1fd5cf5ebb0e454092a19ce",
                subfolder="training",
                dest=LABELS_FOLDER_RASTER,
            )

    # Create class label files by downloading and splitting the regular label file
    if (
        Path(LABELS_FOLDER_CLASSES).is_dir() is False
        or set(EXAMPLE_CLASS_LABEL_FILES).issubset(
            [Path(f).name for f in glob(f"{LABELS_FOLDER_CLASSES}/*.json")]
        )
        is False
    ):
        # Download MONIT02 from EMSR801 (contains 2 spatial features from same date)
        downloaded_file = rapid_mapping_geojson_downloader(
            event_id="801",
            aoi="01",
            monitoring_number="02",
            version="v1",
            dest=LABELS_FOLDER_CLASSES,
        )

        rapid_mapping_class_split(downloaded_file)


@pytest.fixture
def process_labels_setup_csv_datetime():
    create_metadata_headers()
    label_files = glob(f"{LABELS_FOLDER}/*.json")
    for file in label_files:
        strip_date_filename = file.split("/")[-1].replace(
            f"_{file.split('_')[-1]}", ".json"
        )
        shutil.copy(file, f"{LABELS_FOLDER_CSV_DATETIME}/{strip_date_filename}")
        create_metadata_csv(strip_date_filename, file)


@pytest.fixture
def process_labels_setup_csv_datetime_missing_headers():
    os.makedirs(LABELS_FOLDER_CSV_DATETIME, exist_ok=True)
    with open(f"{LABELS_FOLDER_CSV_DATETIME}/metadata.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([["file_name", "not the right headers"]])
    label_files = glob(f"{LABELS_FOLDER}/*.json")
    for file in label_files:
        strip_date_filename = file.split("/")[-1].replace(
            f"_{file.split('_')[-1]}", ".json"
        )
        shutil.copy(file, f"{LABELS_FOLDER_CSV_DATETIME}/{strip_date_filename}")
        create_metadata_csv(strip_date_filename, file)


@pytest.fixture
def process_labels_setup_csv_datetime_invalid_date():
    create_metadata_headers()
    label_files = glob(f"{LABELS_FOLDER}/*.json")
    for file in label_files:
        strip_date_filename = file.split("/")[-1].replace(
            f"_{file.split('_')[-1]}", ".json"
        )
        shutil.copy(file, f"{LABELS_FOLDER_CSV_DATETIME}/{strip_date_filename}")
    create_metadata_csv_invalid_date()


@pytest.fixture(scope="class")
def create_working_dir():
    Path(WORKING_DIR).mkdir(parents=True, exist_ok=True)
    yield
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)


@pytest.fixture
def clean_up_working_dir_contents():
    yield
    if os.path.exists(WORKING_DIR):
        for item in Path(WORKING_DIR).iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)


@pytest.fixture
def process_labels_clean_up_working_dir():
    yield
    working_dir = WORKING_DIR
    print(f"Test clean up. Deleting {working_dir}")
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)


@pytest.fixture
def process_labels_clean_up_default_working_dir():
    yield
    working_dir = DEFAULT_WORKING_DIR
    print(f"Test clean up. Deleting {working_dir}")
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)


@pytest.fixture
def process_labels_clean_up_labels_dir():
    yield
    labels_dir = EMPTY_LABELS_FOLDER
    print(f"Test clean up. Deleting {labels_dir}")
    if os.path.exists(labels_dir):
        shutil.rmtree(labels_dir)


@pytest.fixture
def process_labels_clean_up_csv_labels_dir():
    yield
    labels_dir = LABELS_FOLDER_CSV_DATETIME
    print(f"Test clean up. Deleting {labels_dir}")
    if os.path.exists(labels_dir):
        shutil.rmtree(labels_dir)


def create_metadata_csv(strip_date_filename, file):
    with open(f"{LABELS_FOLDER_CSV_DATETIME}/metadata.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(
            [
                [
                    strip_date_filename,
                    file.split("/")[-1].split("_")[-1].split(".")[0],
                ]
            ]
        )


def create_metadata_csv_invalid_date():
    create_metadata_headers()
    metadata = [
        ["EMSR748_AOI01_DEL_MONIT05_observedEventA_v1.json", "2024-08-26"],
        ["EMSR801_AOI01_DEL_MONIT02_observedEventA_v1.json", "2025/04/23"],
    ]
    with open(f"{LABELS_FOLDER_CSV_DATETIME}/metadata.csv", "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(metadata)


def create_metadata_headers():
    os.makedirs(LABELS_FOLDER_CSV_DATETIME, exist_ok=True)
    with open(f"{LABELS_FOLDER_CSV_DATETIME}/metadata.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([["filename", "date"]])
