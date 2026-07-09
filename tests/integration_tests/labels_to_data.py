# © Copyright IBM Corporation 2025-2026
# SPDX-License-Identifier: Apache-2.0


import os
import pandas as pd
import rasterio
import shutil

from glob import glob
from pathlib import Path

from terrakit.chip.tiling import chip_and_label_data
from terrakit.download.download_data import download_data

from terrakit.store.taco import taco_store_data
from terrakit.transform.labels import process_labels
from terrakit.general_utils.statistics import (
    compute_stats,
    compute_stats_for_masked_pixels,
    load_verified_stats,
)
from terrakit.general_utils.labels_downloader import (
    rapid_mapping_geojson_downloader,
    EXAMPLE_LABEL_FILES,
)

DATASET_NAME = "test_dataset"
WORKING_DIR = f"./tests/resources/intergration_test_data/{DATASET_NAME}"


# Setup
print(f"Test setup up. Deleting {WORKING_DIR}")
if os.path.exists(WORKING_DIR):
    shutil.rmtree(WORKING_DIR)

# Download example labels if these do not already exist:
example_labels_dir = "docs/examples/test_wildfire_vector/"
if (
    Path(example_labels_dir).is_dir() is False
    or set(EXAMPLE_LABEL_FILES).issubset(glob(f"{example_labels_dir}/*.json")) is False
):
    rapid_mapping_geojson_downloader(
        event_id="748",
        aoi="01",
        monitoring_number="05",
        version="v1",
        dest="docs/examples/test_wildfire_vector",
    )
    rapid_mapping_geojson_downloader(
        event_id="801",
        aoi="01",
        monitoring_number="02",
        version="v1",
        dest="docs/examples/test_wildfire_vector",
    )

# Complete
print("End...\n\n")


####################################################################
#                                                                  #
# ## 1. Process labels                                             #
#                                                                  #
####################################################################
#                                                                  #
print("\n\nTEST 1.1:....Test process_labels...")  #
#                                                                  #
####################################################################
labels_folder = "./docs/examples/test_wildfire_vector"
labels_gdf, grouped_bbox_gdf = process_labels(
    dataset_name=DATASET_NAME,
    working_dir=WORKING_DIR,
    labels_folder=labels_folder,
)
# Validate
assert isinstance(grouped_bbox_gdf, pd.DataFrame)
assert "datetime" in list(grouped_bbox_gdf.columns.values)
assert "test_dataset_all_bboxes.shp" in os.listdir(WORKING_DIR)
assert "geometry" in list(labels_gdf.columns.values)
assert "test_dataset_labels.shp" in os.listdir(WORKING_DIR)

# Completed
print("End....\n\n")


####################################################################
#                                                                  #
print("\n\nTEST 2.1:....Download the data")  #
#                                                                  #
####################################################################
# Use a shp file to download data
config = {
    "download": {
        "data_sources": [
            {
                "data_connector": "sentinel_aws",
                "collection_name": "sentinel-2-l2a",
                "bands": ["blue", "green", "red"],
                "save_file": "",
            },
        ],
        "date_allowance": {"pre_days": 0, "post_days": 21},
        "max_cloud_cover": 10,
        "transform": {
            "scale_data_xarray": True,
            "impute_nans": True,
            "reproject": True,
        },
    },
}

queried_data = download_data(
    dataset_name=DATASET_NAME,
    data_sources=config["download"]["data_sources"],
    date_allowance=config["download"]["date_allowance"],
    transform=config["download"]["transform"],
    working_dir=WORKING_DIR,
    keep_files=False,
)
# Check queried data is come back as with len(queried_data) > 0
assert len(queried_data) > 0

####################################################################
#                                                                  #
print("\n\nTEST 3.1... Chip the data")  #
#                                                                  #
####################################################################
# Now that the tiled data has been downloaded, let's chip it accordingly.

res = chip_and_label_data(
    dataset_name=DATASET_NAME,
    sample_dim=256,
    queried_data=queried_data,
    keep_files=True,
)
assert len(res) > 0

# Clean up and re-run using working dir
chips = glob(f"{WORKING_DIR}/*.data.tif")
labels = glob(f"{WORKING_DIR}/*.label.tif")
for chip in chips:
    os.remove(chip)

for label in labels:
    os.remove(label)

# Run again using working dir
res = chip_and_label_data(
    dataset_name=DATASET_NAME,
    sample_dim=256,
    working_dir=WORKING_DIR,
    keep_files=False,
)
assert len(res) > 0
####################################################################
#                                                                  #
# # ## 4. Store                                                    #
#                                                                  #
####################################################################

taco_store_data(
    dataset_name=DATASET_NAME,
    working_dir=WORKING_DIR,
    license="CC-BY 4.0",
    save_dir=WORKING_DIR,
)
assert f"{DATASET_NAME}.tacozip" in os.listdir(WORKING_DIR)

####################################################################
#                                                                  #
# # ## 5. Upload                                                   #
#                                                                  #
####################################################################
# TODO

####################################################################
#                                                                  #
print("\n\nTEST 6.1... Dataset factory integration test")  #
#                                                                  #
####################################################################
# Confirm labels and data files are correctly names with appropriate file suffixes.
file_suffix = "data.tif"
label_suffix = "label.tif"
files = sorted(
    glob(
        WORKING_DIR + "/**/*" + file_suffix,
        recursive=True,
    )
)
assert len(files) > 0

label_files = sorted(
    glob(
        WORKING_DIR + "/**/*" + label_suffix,
        recursive=True,
    )
)
assert len(label_files) > 0

image_stems = [filepath.replace(file_suffix, "").split("/")[-1] for filepath in files]
label_stems = [
    filepath.replace(label_suffix, "").split("/")[-1] for filepath in label_files
]

assert image_stems == label_stems


####################################################################
#                                                                  #
print("\n\nTEST 7.1... Verify label and data statistics")  #
#                                                                  #
####################################################################

# Verify statistics
print("\n.\n.\n.\n.\n.\nVerifying statistics....\n.")
target_tif = "sentinel_aws_sentinel-2-l2a_2024-08-30_imputed_20"
verified_label_stats, verified_data_stats, verified_mask_stats = load_verified_stats()

# Check the corresponding labels file that has just been generated has the same stats.
print("Validating labels...")
with rasterio.open(f"{WORKING_DIR}/{target_tif}.label.tif") as src_mask:
    target_mask = src_mask.read(1)
target_label_stats = [mean_val, median_val, min_val, max_val, std_dev, count] = (
    compute_stats(target_mask)
)
assert verified_label_stats == target_label_stats

# Check the corresponding data file that has just been generated has the same stats.
print("Validating data...")
with rasterio.open(f"{WORKING_DIR}/{target_tif}.data.tif") as src_mask:
    target_data = src_mask.read(1)
target_data_stats = [mean_val, median_val, min_val, max_val, std_dev, count] = (
    compute_stats(target_data)
)
assert verified_data_stats == target_data_stats

# Check the corresponding data/label mask generate the same stats.
print("Validating mask...")
target_mask_stats = [mean_val, median_val, min_val, max_val, std_dev, count] = (
    compute_stats_for_masked_pixels(target_data, target_mask)
)
assert verified_mask_stats == target_mask_stats


####################################################################
#                                                                  #
print(f"\n\nTest clean up. Deleting {WORKING_DIR}")  #
#                                                                  #
####################################################################
if os.path.exists(WORKING_DIR):
    shutil.rmtree(WORKING_DIR)

# Complete
print("Tests passed...\n\n")
