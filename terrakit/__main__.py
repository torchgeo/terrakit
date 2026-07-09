# © Copyright IBM Corporation 2025-2026
# SPDX-License-Identifier: Apache-2.0


import logging
import sys

from jsonargparse import ArgumentParser

from terrakit.download.download_data import DownloadCls, download_data
from terrakit.chip.tiling import ChipAndLabelCls, chip_and_label_data
from terrakit.store.taco import TacoCls, taco_store_data
from terrakit.transform.labels import LabelsCls, process_labels
from terrakit.validate.pipeline_model import PipelineModel
from terrakit._version import VERSION, PACKAGE_NAME

logger = logging.getLogger(__name__)


def main() -> None:
    if len(sys.argv) == 1:
        print(
            "usage: terrakit [-h] [--dataset_name DATASET_NAME] [--working_dir WORKING_DIR] [--config CONFIG] [--print_config[=flags]] {labels,download,chip,store,upload}...\n"
        )
        exit(0)

    # Create labels parser
    parser_labels = ArgumentParser()
    parser_labels.add_class_arguments(LabelsCls)

    # Create download parser
    parser_download = ArgumentParser()
    parser_download.add_class_arguments(DownloadCls)

    parser_chip = ArgumentParser()
    parser_chip.add_class_arguments(ChipAndLabelCls)

    parser_store = ArgumentParser()
    parser_store.add_class_arguments(TacoCls)

    parser_upload = ArgumentParser()
    parser_upload.add_argument("--op5")

    parser = ArgumentParser(prog=PACKAGE_NAME, version=VERSION)
    parser.add_class_arguments(PipelineModel)

    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("labels", parser_labels)
    subcommands.add_subcommand("download", parser_download)
    subcommands.add_subcommand("chip", parser_chip)
    subcommands.add_subcommand("store", parser_store)
    subcommands.add_subcommand("upload", parser_upload)

    parser.add_argument("--config", action="config")

    cfg = parser.parse_args()
    if cfg.subcommand == "labels":
        process_labels(
            dataset_name=cfg.dataset_name,
            working_dir=cfg.working_dir,
            active=cfg.labels.active,
            labels_folder=cfg.labels.labels_folder,
            label_type=cfg.labels.label_type,
            datetime_info=cfg.labels.datetime_info,
        )
    elif cfg.subcommand == "download":
        download_data(
            dataset_name=cfg.dataset_name,
            working_dir=cfg.working_dir,
            date_allowance=cfg.download.date_allowance,
            transform=cfg.download.transform,
            data_sources=cfg.download.data_sources,
            active=cfg.download.active,
            max_cloud_cover=cfg.download.max_cloud_cover,
            datetime_bbox_shp_file=cfg.download.datetime_bbox_shp_file,
            labels_shp_file=cfg.download.labels_shp_file,
            keep_files=cfg.download.keep_files,
        )
    elif cfg.subcommand == "chip":
        chip_and_label_data(
            dataset_name=cfg.dataset_name,
            working_dir=cfg.working_dir,
            active=cfg.chip.active,
            data_suffix=cfg.chip.data_suffix,
            label_suffix=cfg.chip.label_suffix,
            chip_suffix=cfg.chip.chip_suffix,
            chip_label_suffix=cfg.chip.chip_label_suffix,
            sample_dim=cfg.chip.sample_dim,
        )
    elif cfg.subcommand == "store":
        taco_store_data(
            cfg.dataset_name,
            working_dir=cfg.working_dir,
            license=cfg.store.license,
            active=cfg.store.active,
            format=cfg.store.format,
            save_dir=cfg.store.save_dir,
            tortilla_name=cfg.store.tortilla_name,
        )
    elif cfg.subcommand == "upload":
        print("TODO: Upload labels..")
    else:
        print(
            "Error: Missing subcommand. - Please specify one of {labels,download,chip,store,upload}."
        )


if __name__ == "__main__":
    main()
