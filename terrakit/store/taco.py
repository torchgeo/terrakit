# © Copyright IBM Corporation 2025-2026
# SPDX-License-Identifier: Apache-2.0


import json
import logging
import rasterio
import re
import tacoreader
import tacotoolbox

from datetime import datetime
from pathlib import Path
from tacotoolbox.datamodel import Sample, Tortilla
from tacotoolbox.sample.extensions.stac import STAC
from tacotoolbox.taco.datamodel import Extent, Provider, Taco
from tacotoolbox._exceptions import TacoValidationError
from typing import Any, List, Optional, Tuple

from terrakit.general_utils.curation_metadata import dataset_metdata
from terrakit.general_utils.exceptions import (
    TerrakitValidationError,
    TerrakitValueError,
)
from terrakit.validate.pipeline_model import pipeline_model_validation
from terrakit.validate.store_model import StoreModel

logger = logging.getLogger(__name__)


# TacoStoreConfig is now replaced by StoreModel from terrakit.validate.store_model
# Keeping this alias for backward compatibility
TacoStoreConfig = StoreModel


class TacoCls:
    def __init__(
        self,
        *,
        active: bool,
        license: str,
        format: str = "taco",
        save_dir: str = "./tmp",
        tortilla_name: str = "",
        statistics: bool = True,
        include_config: bool = True,
        check_dataset: bool = True,
    ):
        """
        Args:
            active: Set the set to active or inactive
            license: License name for the dataset (e.g., 'CC-BY-4.0')
            type: Either "taco" or "local"
            save_dir: Directory to save the tortilla files
            tortilla_name: Name of the final tortilla file
        """
        self.active = active
        self.license = license
        self.format = format
        self.save_dir = save_dir
        self.tortilla_name = tortilla_name
        self.statistics = statistics
        self.include_config = include_config
        self.check_dataset = check_dataset

    def _extract_date_from_path(self, path: Path) -> str:
        match = re.search(r"\d{4}-\d{2}-\d{2}", path.name)
        if match is None:
            raise ValueError(f"Could not extract date from filename: {path}")
        return match.group()

    def _to_microseconds(self, timestamp: datetime) -> int:
        return int(timestamp.timestamp() * 1_000_000)

    def _build_stac_metadata(self, raster_path: Path, sample_id: str) -> STAC:
        with rasterio.open(raster_path) as src:
            start_date = self._extract_date_from_path(raster_path)
            start_dt = datetime.fromisoformat(f"{start_date}T00:00:00+00:00")
            end_dt = datetime.fromisoformat(f"{start_date}T23:59:59+00:00")
            return STAC(
                sample_id=sample_id,
                crs=str(src.crs),
                tensor_shape=(src.count, src.height, src.width),
                geotransform=src.transform.to_gdal(),
                time_start=self._to_microseconds(start_dt),
                time_end=self._to_microseconds(end_dt),
            )

    def _build_extent(self, raster_path: Path) -> Extent:
        with rasterio.open(raster_path) as src:
            bounds = src.bounds
            date_str = self._extract_date_from_path(raster_path)
            return Extent(
                spatial=[bounds.left, bounds.bottom, bounds.right, bounds.top],
                temporal=[
                    f"{date_str}T00:00:00Z",
                    f"{date_str}T23:59:59Z",
                ],
            )

    def _find_file_pairs(
        self,
        directory: Path,
        data_suffix: str = ".data.tif",
        label_suffix: str = ".label.tif",
    ) -> List[Tuple[Path, Path]]:
        """Find all matching .data.tif and .label.tif file pairs in a directory."""
        data_files: list[Path] = sorted(directory.glob("*" + data_suffix))
        pairs: list[Any] = []

        for data_file in data_files:
            # Extract the base name by removing .data.tif suffix
            base_name: str = data_file.name.replace(data_suffix, "")
            label_file: Path = directory / f"{base_name}{label_suffix}"

            if label_file.exists():
                pairs.append((data_file, label_file))
            else:
                print(f"Warning: No matching label file found for {data_file.name}")

        return pairs

    def _combine_extents(self, extents: List[Extent]) -> Extent:
        """Combine multiple extents into a single extent covering all."""
        if not extents:
            raise ValueError("No extents to combine")

        # Combine spatial extents (find bounding box)
        all_spatial = [e.spatial for e in extents]
        min_x = min(bbox[0] for bbox in all_spatial)
        min_y = min(bbox[1] for bbox in all_spatial)
        max_x = max(bbox[2] for bbox in all_spatial)
        max_y = max(bbox[3] for bbox in all_spatial)

        # Combine temporal extents (find earliest and latest)
        all_temporal = [e.temporal for e in extents]
        earliest = min(temp[0] for temp in all_temporal)
        latest = max(temp[1] for temp in all_temporal)

        return Extent(
            spatial=[min_x, min_y, max_x, max_y],
            temporal=[earliest, latest],
        )

    def create_tortilla(
        self,
        dataset_name: str,
        working_dir: str = "./tmp",
        save_dir: Optional[str] = None,
        chip_suffix: str = ".data.tif",
        label_suffix: str = ".label.tif",
    ) -> str:
        """Create a taco dataset from all matching data and label files in a directory."""
        data_label_dir: Path = Path(working_dir)

        resolved_save_dir = (
            Path(save_dir) if save_dir is not None else Path(self.save_dir)
        )
        output_path = (
            Path(self.tortilla_name)
            if self.tortilla_name
            else Path(f"{dataset_name}.tacozip")
        )
        if not output_path.is_absolute():
            output_path = resolved_save_dir / output_path
        output_path = output_path.with_suffix(".tacozip")

        file_pairs: List[Tuple[Path, Path]] = self._find_file_pairs(
            directory=data_label_dir, data_suffix=chip_suffix, label_suffix=label_suffix
        )

        if not file_pairs:
            raise TerrakitValueError(
                f"No {chip_suffix}/{label_suffix} file pairs found in {working_dir}"
            )
        logger.info(f"Found {len(file_pairs)} {chip_suffix}/{label_suffix} file pairs")

        samples: list[Any] = []
        extents: list[Any] = []

        for data_path, label_path in file_pairs:
            logger.debug("Processing data file: %s", data_path)
            # Create unique sample IDs based on filenames
            data_id = data_path.name  # Removes .data.tif
            label_id = label_path.name  # Removes .label.tif

            # Build STAC metadata for data
            stac_metadata = self._build_stac_metadata(
                raster_path=data_path, sample_id=data_id
            )

            # Create data sample
            data_sample = Sample(
                id=data_id,
                path=data_path,
                type="FILE",
            )
            data_sample.extend_with(stac_metadata)
            samples.append(data_sample)

            # Create label sample
            label_sample = Sample(
                id=label_id,
                path=label_path,
                type="FILE",
            )
            label_sample.extend_with(
                stac_metadata.model_copy(update={"sample_id": label_id})
            )
            samples.append(label_sample)

            # Collect extent for this pair
            extents.append(self._build_extent(data_path))

        # logger.info(f"Creating taco from samples: {', '.join(s.path for s in samples)}")
        print(samples[0])
        Tortilla(samples=samples)
        # Combine extents from all samples
        combined_extent = self._combine_extents(extents)

        taco = Taco(
            tortilla=Tortilla(samples=samples),
            id=output_path.stem,
            dataset_version="1.0.0",
            description=f"Multi-sample taco dataset with {len(file_pairs)} samples",
            licenses=[self.license],
            extent=combined_extent,
            providers=[Provider(name="TerraKit")],
            tasks=["other"],
        )

        try:
            tacotoolbox.create(taco, output_path)
        except TacoValidationError as err:
            raise (TerrakitValidationError(f"Failed to create taco: {err}"))

        logger.info(f"Created taco dataset at {output_path}")
        return str(output_path)


def taco_store_data(
    dataset_name: str,
    license: str,
    working_dir: str = "./tmp",
    active: bool = True,
    format: str = "taco",
    save_dir: str = "./tmp",
    tortilla_name: str = "",
    statistics: bool = True,
    include_config: bool = True,
    check_dataset: bool = True,
) -> str:
    """
    Main function to store Taco dataset data.

    This function initializes the Taco class with chip-specific arguments, validates the Taco model,
    and creates a tortilla (stores data) using the provided dataset name and working directory.

    Args:
        dataset_name: Name of the dataset
        working_dir: Directory containing the data files
        license: License name for the dataset (e.g., 'CC-BY-4.0'). Required.
        active: Set the set to active or inactive
        format: Either "taco" or "local"
        save_dir: Directory to save the tortilla files
        tortilla_name: Name of the final tortilla file
        statistics: Whether to include statistics
        include_config: Whether to include configuration
        check_dataset: Whether to check the dataset

    Returns:
        Path to the created tortilla file

    Raises:
        TerrakitValidationError
    """
    pipeline_model = pipeline_model_validation(
        dataset_name=dataset_name, working_dir=working_dir
    )

    taco = TacoCls(
        active=active,
        license=license,
        format=format,
        save_dir=save_dir,
        tortilla_name=tortilla_name,
        statistics=statistics,
        include_config=include_config,
        check_dataset=check_dataset,
    )  # Initialize class with chip specific args
    taco_model = TacoStoreConfig.model_validate(
        taco
    )  # validate store model - do this in the store class
    logging.info(f"Storing data with arguments: {taco_model}")
    store_data = taco.create_tortilla(dataset_name, working_dir, save_dir)

    # Save dataset metadata to file
    store_metadata = {
        "step_id": "store",
        "activity": "Package dataset in taco format",
        "method": "terrakit.store.taco.taco_store_data",
        "working_dir": str(working_dir),
        "parameters": json.loads(taco_model.model_dump_json()),
    }
    dataset_metdata(pipeline_model, store_metadata)

    return store_data


def load_taco(tortilla_name):
    """Load a taco dataset and return it as a dataframe.

    Args:
        tortilla_name: Path to the taco dataset file

    Returns:
        DataFrame containing the taco dataset

    Raises:
        TerrakitValueError: If the taco dataset cannot be loaded
    """
    try:
        ds = tacoreader.load(tortilla_name)
        logger.info(f"ID: {ds.id}")
        logger.info(f"Version: {ds.version}")
        logger.info(msg=f"Samples: {len(ds.data)}")
        return ds
    except Exception as e:
        raise TerrakitValueError(
            f"Failed to load taco dataset from {tortilla_name}: {e}"
        )
