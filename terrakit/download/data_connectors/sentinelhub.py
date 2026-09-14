# © Copyright IBM Corporation 2025-2026
# SPDX-License-Identifier: Apache-2.0


# Assisted by watsonx Code Assistant

import os
import json
import numpy as np
import xarray as xr
import rioxarray
import shutil
from string import Template  # noqa
import glob
from datetime import datetime
import logging

from oauthlib.oauth2.rfc6749.errors import InvalidClientError  # type: ignore[import-untyped]
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,  # noqa
    MimeType,
    MosaickingOrder,  # noqa
    SHConfig,
    SentinelHubCatalog,
    SentinelHubRequest,
    bbox_to_dimensions,
)
from typing import Any

from ..geodata_utils import (
    load_and_list_collections,
    save_data_array_to_file,
)
from ...general_utils.exceptions import (
    TerrakitValidationError,
    TerrakitValueError,
    TerrakitNoDataFoundError,
)
from ..connector import Connector
from terrakit.validate.helpers import (
    check_collection_exists,
    check_start_end_date,
    check_bbox,
    check_area_polygon,
)

logger = logging.getLogger(__name__)

######################################################################################################
###  Supporting functions
######################################################################################################


def get_sh_config():
    config = None
    sh_client_id = os.getenv("SH_CLIENT_ID")
    sh_client_secret = os.getenv("SH_CLIENT_SECRET")
    client_id = sh_client_id
    client_secret = sh_client_secret
    sh_token_url = "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"
    sh_base_url = "https://services.sentinel-hub.com"
    try:
        config = SHConfig(
            profile="cdse",
            sh_client_id=client_id,
            sh_client_secret=client_secret,
            sh_base_url=sh_base_url,
            sh_token_url=sh_token_url,
        )
    except Exception as exc:
        logger.error(
            f"Something wrong happened with sentinel hub connection based on config, trace: {exc}"
        )
        config = SHConfig(
            sh_client_id=client_id,
            sh_client_secret=client_secret,
            sh_base_url=sh_base_url,
            sh_token_url=sh_token_url,
        )

    return config


def create_request(
    sh_config,
    data_details,
    bands,
    aoi_bbox,
    aoi_size,
    timestamp_start,
    timestamp_end,
    data_folder="./",
    maxcc=None,
):
    qt = data_details["query_template"]
    if not qt.strip().startswith("//VERSION=3"):
        raise TerrakitValidationError(
            "Invalid query_template value: does not start with //VERSION=3"
        )
    evalscript = Template(qt).substitute(
        {
            "bands": str(bands),
            "num_bands": len(bands),
            "band_samples": str(["sample." + b for b in bands]).replace("'", ""),
        }
    )

    dc_name = data_details["data_collection"]  # e.g. "DataCollection.SENTINEL2_L1C"
    dc_prefix = "DataCollection."
    if not dc_name.startswith(dc_prefix):
        raise TerrakitValidationError(f"Invalid data_collection value: {dc_name!r}")
    member_name = dc_name[len(dc_prefix) :]
    data_collection = getattr(DataCollection, member_name)

    shr = SentinelHubRequest.input_data(
        data_collection=data_collection,
        time_interval=(timestamp_start, timestamp_end),
    )

    if maxcc is not None:
        shr["maxCloudCoverage"] = maxcc / 100.0
    if "mosaicking_order" in data_details["request_input_data"]:
        mosaicking_order_name = data_details["request_input_data"][
            "mosaicking_order"
        ]  # i.e. "MosaickingOrder.LEAST_CC"
        mosaicking_order_prefix = "MosaickingOrder."
        if not mosaicking_order_name.startswith(mosaicking_order_prefix):
            raise TerrakitValidationError(
                f"Invalid mosaicking_order value: {mosaicking_order_name!r}"
            )
        member_name = mosaicking_order_name[len(mosaicking_order_prefix) :]
        shr["mosaickingOrder"] = getattr(MosaickingOrder, member_name).value

    logger.info(shr)

    sh_request = SentinelHubRequest(
        data_folder=data_folder,
        evalscript=evalscript,
        input_data=[shr],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=aoi_bbox,
        size=aoi_size,
        config=sh_config,
    )

    return sh_request


def sh_get_data(
    sh_config,
    collection_details,
    bbox,
    data_date,
    band_names,
    save_file,
    sh_data_dir="./sh_data",
):
    ###  Create folder for individual date data download
    shutil.rmtree(sh_data_dir, ignore_errors=True)
    os.makedirs(sh_data_dir, exist_ok=True)

    # bands_dict = ds_spec["bands"]
    # band_names = list(bands_dict.values())

    aoi_bbox = BBox(bbox, crs=CRS.WGS84)
    aoi_size = bbox_to_dimensions(
        bbox=aoi_bbox, resolution=collection_details["resolution_m"]
    )

    # logger.info(aoi_bbox,aoi_size)

    sh_request = create_request(
        sh_config,
        collection_details,
        band_names,
        aoi_bbox,
        aoi_size,
        data_date,
        data_date,
        data_folder=sh_data_dir,
        maxcc=80,
    )
    logger.info(sh_request.payload)
    sh_request.save_data()

    file_dir = glob.glob(f"{sh_data_dir}/*/*.tiff")
    if len(file_dir) == 0:
        logger.info(f"No data downloaded for {data_date}")
    elif len(file_dir) > 1:
        logger.info(f"Multiple files downloaded for {data_date}, using first one")
    file_dir = file_dir[0]

    da_list = []

    da = rioxarray.open_rasterio(file_dir)
    if not (da.values == 0).all():
        # Ensure date with data
        data_date_datetime = datetime.strptime(data_date, "%Y-%m-%d")
        da = da.assign_coords({"band": band_names, "time": data_date_datetime})
        da_list.append(da)

        # logger.debug(f'---->>>---->>>>---- {da_list}')
        da = xr.concat(da_list, dim="time")

        # Replace no data with -9999
        nan = xr.DataArray([0] * da.sizes["band"], dims=["band"])
        da = xr.where(da == nan, -9999, da)
        da.assign_attrs(_FillValue=-9999)

        # Convert to float32
        da = da.astype("float32")

        da = da.fillna(-9999.0)
        da = da.rio.write_crs("EPSG:4326")
        da.rio.write_nodata(-9999, inplace=True)

    return da


######################################################################################################
###  Connector class
######################################################################################################


class SentinelHub(Connector):
    """
    A class to interact with Sentinel Hub data services.

    Attributes:
        collections (list): A list of available collections.
        collections_details (list): Detailed information about the collections.
        sh_config (SHConfig): Configuration settings for Sentinel Hub.
    """

    def __init__(self):
        """
        Initialize SentinelHub with collections and configuration.
        """
        self.connector_type = "sentinelhub"
        self.collections: list[Any] = load_and_list_collections(
            connector_type="sentinelhub"
        )
        self.collections_details = load_and_list_collections(
            as_json=True, connector_type="sentinelhub"
        )
        self.sh_config = SHConfig()

    def list_collections(self) -> list[Any]:
        """
        Lists the available collections.

        Returns:
            list: A list of collection names.
        """
        logger.info("Listing available collections")
        return self.collections

    def find_data(
        self,
        data_collection_name: str,
        date_start: str,
        date_end: str,
        area_polygon=None,
        bbox=None,
        bands=[],
        maxcc=100,
        data_connector_spec=None,
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """
        This function retrieves unique dates and corresponding data results from a specified Sentinel Hub data collection.

        Parameters:
            data_collection_name (str): The name of the Sentinel Hub data collection to search.
            date_start (str): The start date for the time interval in 'YYYY-MM-DD' format.
            date_end (str): The end date for the time interval in 'YYYY-MM-DD' format.
            area_polygon (Polygon, optional): A polygon defining the area of interest.
            bbox (tuple, optional): A bounding box defining the area of interest in the format (minx, miny, maxx, maxy).
            bands (list, optional): A list of bands to retrieve. Defaults to [].
            maxcc (int, optional): The maximum cloud cover percentage for the data. Default is 100 (no cloud cover filter).
            data_connector_spec (list, optional): A dictionary containing the data connector specification.

        Returns:
            tuple: A tuple containing a sorted list of unique dates and a list of data results.

        Raises:
            TerrakitValidationError: If a validation error occurs.
            TerrakitValueError: If a value error occurs.
            TerrakitNoDataFoundError: If no files are found for the requested query.
        """
        # Check credentials have been set correctly.
        if "SH_CLIENT_ID" not in os.environ and "SH_CLIENT_SECRET" not in os.environ:
            raise TerrakitValidationError(
                message="Error: Missing credentials 'SH_CLIENT_ID' and 'SH_CLIENT_SECRET'. Please update .env with correct credentials."
            )

        # Check data_collection_name exists in self.collections.
        check_collection_exists(data_collection_name, self.collections)

        # Check date_start and date_end are in the correct format.
        check_start_end_date(date_start=date_start, date_end=date_end)
        check_area_polygon(
            area_polygon=area_polygon, connector_type=self.connector_type
        )
        check_bbox(bbox=bbox, connector_type=self.connector_type)

        if data_connector_spec is None:
            data_connector_spec_list = [
                X
                for X in self.collections_details
                if X["collection_name"] == data_collection_name
            ]
            if len(data_connector_spec_list) == 0:
                error_msg = (
                    f"Unable to find collection details for '{data_collection_name}'"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            data_connector_spec = data_connector_spec_list[0]

        dc_name = data_connector_spec[
            "data_collection"
        ]  # e.g. "DataCollection.SENTINEL2_L1C"
        dc_prefix = "DataCollection."
        if not dc_name.startswith(dc_prefix):
            raise TerrakitValidationError(f"Invalid data_collection value: {dc_name!r}")
        member_name = dc_name[len(dc_prefix) :]
        data_collection = getattr(DataCollection, member_name)

        self.sh_config.sh_base_url = data_collection.service_url
        logger.info(self.sh_config.sh_base_url)
        dataset_catalog = SentinelHubCatalog(config=self.sh_config)

        if "filter" in data_connector_spec["search"]:
            filter_string = data_connector_spec["search"]["filter"]

            for X in data_connector_spec["request_input_data"]:
                if X == "maxcc":
                    filter_string = filter_string.replace(X, str(maxcc))
                else:
                    filter_string = filter_string.replace(
                        X, str(data_connector_spec["request_input_data"][X])
                    )
        else:
            filter_string = ""

        if "fields" in data_connector_spec["search"]:
            fields_dict = json.loads(data_connector_spec["search"]["fields"])

        else:
            fields_dict = {"include": ["id", "properties.datetime"], "exclude": []}

        time_interval = date_start, date_end

        if bbox is not None:
            aoi_bbox = BBox(bbox=bbox, crs=CRS.WGS84)

            search_iterator = dataset_catalog.search(
                data_collection,
                bbox=aoi_bbox,
                time=time_interval,
                filter=filter_string,
                fields=fields_dict,
            )
        elif area_polygon is not None:
            search_iterator = dataset_catalog.search(
                data_collection,
                intersects=area_polygon,
                time=time_interval,
                filter=filter_string,
                fields=fields_dict,
            )
        else:
            error_msg = f"Error: Issue finding data from {self.connector_type}. Please specify at least one of 'bbox' and 'area_polygon'"
            logger.error(error_msg)
            raise TerrakitValueError(error_msg)

        try:
            results = list(search_iterator)
        except InvalidClientError as e:
            error_msg = (
                f"Error: Issue authenticating. Check credentials are up to date.{e}"
            )
            logger.error(error_msg)
            raise TerrakitValidationError(error_msg)

        if len(results) == 0:
            error_msg = (
                f"No data found for collection '{data_collection_name}' "
                f"between {date_start} and {date_end}."
            )
            logger.warning(error_msg)
            raise TerrakitNoDataFoundError(error_msg)

        # Log average cloud cover if maxcc filtering was applied
        if maxcc < 100 and results:
            cloud_covers = [
                X["properties"].get("eo:cloud_cover", 0)
                for X in results
                if "eo:cloud_cover" in X.get("properties", {})
            ]
            if cloud_covers:
                average_cloud_cover = np.mean(cloud_covers)
                logger.info(f"Average cloud cover: {average_cloud_cover}%")

        unique_dates = sorted(set([X["properties"]["datetime"][0:10] for X in results]))
        return unique_dates, results

    def get_data(
        self,
        data_collection_name,
        date_start,
        date_end,
        area_polygon=None,
        bbox=None,
        bands=[],
        maxcc=100,
        data_connector_spec=None,
        save_file=None,
        working_dir=".",
    ) -> xr.DataArray:
        """
        Fetches data from SentinelHub for the specified collection, date range, area, and bands.

        Parameters:
            data_collection_name (str): Name of the data collection to fetch data from.
            date_start (str): Start date for the data retrieval (inclusive), in 'YYYY-MM-DD' format.
            date_end (str): End date for the data retrieval (inclusive), in 'YYYY-MM-DD' format.
            area_polygon (list, optional): Polygon defining the area of interest. Defaults to None.
            bbox (list, optional): Bounding box defining the area of interest. Defaults to None.
            bands (list, optional): List of bands to retrieve. Defaults to all bands.
            maxcc (int, optional): Maximum cloud cover threshold (0-100). Defaults to 100.
            data_connector_spec (dict, optional): Data connector specification. Defaults to None.
            save_file (str, optional): Path to save the output file. If provided, individual GeoTIFF files
                will be saved for each date with the naming pattern: {save_file}_{date}.tif. Each file
                contains all requested bands for that specific date. If None, no files are saved to disk. Defaults to None.
            working_dir (str, optional): Working directory for temporary files. Defaults to '.'.

        Returns:
            xarray.DataArray: An xarray DataArray containing all fetched data with dimensions (time, band, y, x).
                All dates are stacked along the time dimension, and all bands are stacked along the band dimension.
                If save_file is provided, individual date files are also saved to disk.

        Raises:
            TerrakitValidationError: If a validation error occurs.
            TerrakitValueError: If a value error occurs.
            TerrakitNoDataFoundError: If no files are found for the requested query.
        """
        # Check credentials have been set correctly.
        if "SH_CLIENT_ID" not in os.environ and "SH_CLIENT_SECRET" not in os.environ:
            error_msg = "Error: Missing credentials 'SH_CLIENT_ID' and 'SH_CLIENT_SECRET'. Please update .env with correct credentials."
            logger.warning(error_msg)
            raise TerrakitValidationError(error_msg)

        # Check data_collection_name exists in self.collections.
        if data_collection_name not in self.collections:
            error_msg = f"Invalid collection '{data_collection_name}'. Please choose from one of the following collection {self.collections}"
            logger.warning(error_msg)
            raise TerrakitValueError(error_msg)

        logger.info(bands)
        collection_details = [
            X
            for X in self.collections_details
            if X["collection_name"] == data_collection_name
        ][0]

        unique_dates, res = self.find_data(
            data_collection_name, date_start, date_end, bbox=bbox, maxcc=maxcc
        )

        if unique_dates == []:
            error_msg = (
                f"No data found for the specified date range "
                f"{date_start}:{date_end}. Unique dates: {unique_dates}"
            )
            logger.warning(error_msg)
            raise TerrakitNoDataFoundError(error_msg)
        da_list = []
        logger.info(f"The following unique dates were found: {unique_dates}")
        for udate in unique_dates:  # type: ignore[union-attr]
            usave_file = (
                save_file.replace(".tif", f"_{udate}.tif")
                if save_file is not None
                else None
            )

            da: xr.DataArray = sh_get_data(
                self.sh_config,
                collection_details,
                bbox,
                udate,
                bands,
                usave_file,
                sh_data_dir=f"{working_dir}/sh_data",
            )

            if da.size == 0:
                logger.warning("No usable data returned for date %s", udate)
                continue

            da_list.append(da)

        if len(da_list) == 0:
            error_msg = (
                f"No usable data retrieved for collection '{data_collection_name}' "
                f"between {date_start} and {date_end}."
            )
            logger.warning(error_msg)
            raise TerrakitNoDataFoundError(error_msg)

        logger.info("Concatenating data...")
        da = xr.concat(da_list, dim="time")

        # Save to file
        save_data_array_to_file(da, save_file)

        sh_data_dir = f"{working_dir}/sh_data"
        logging.info(f"Removing dir {sh_data_dir}")
        shutil.rmtree(sh_data_dir, ignore_errors=True)

        return da
