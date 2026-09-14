# © Copyright IBM Corporation 2025-2026
# SPDX-License-Identifier: Apache-2.0


import pytest
import xarray as xr

from terrakit import DataConnector
from terrakit.general_utils.exceptions import TerrakitNoDataFoundError


class TestNASAEarthData:
    connector_type = "nasa_earthdata"
    bands = ["B04", "B03", "B02"]

    def test_list_collections_nasa_earthdata(
        self,
        mock_setup_nasa,
        **kwargs,
    ):
        expected_collections = ["HLSS30_2.0", "HLSL30_2.0"]
        dc = DataConnector(connector_type=self.connector_type)
        collections = dc.connector.list_collections()
        assert collections == expected_collections

    def test_list_collection_with_invalid_credentials_nasa_earthdata(
        self,
        **kwargs,
    ):
        """
        Test that list_collection returns as expected, even when invalid credentials are provided.
        """
        pass

    @pytest.mark.parametrize("collection", ["HLSS30_2.0", "HLSL30_2.0"])
    @pytest.mark.parametrize("maxcc", [100, 30])
    def test_find_available_data_nasa_earthdata(
        self,
        mock_nasa_find_datasets,
        collection,
        start_date,
        end_date,
        bbox,
        maxcc,
    ):
        mock_find_items_nasa = mock_nasa_find_datasets
        dc = DataConnector(connector_type=self.connector_type)
        unique_dates, results = dc.connector.find_data(
            data_collection_name=collection,
            date_start=start_date,
            date_end=end_date,
            bbox=bbox,
            bands=self.bands,
            maxcc=maxcc,
        )
        mock_find_items_nasa.assert_called_once()
        match (collection, maxcc):
            case ("HLSS30_2.0", 100):
                assert unique_dates == [
                    "2024-01-06",
                    "2024-01-11",
                    "2024-01-16",
                    "2024-01-26",
                    "2024-01-31",
                ]
                assert len(results) == 10
            case ("HLSS30_2.0", 30):
                assert unique_dates == ["2024-01-06", "2024-01-26", "2024-01-31"]
                assert len(results) == 5

        assert isinstance(results, list)
        assert set(results[0].keys()) == {"id", "properties"}
        assert set(results[0]["properties"].keys()) == {"datetime", "eo:cloud_cover"}

    @pytest.mark.parametrize(
        "collection",
        [
            "HLSL30_2.0",
            "HLSS30_2.0",
        ],
    )
    def test_get_data_nasa_earthdata(
        self,
        mock_nasa_download_datasets,
        collection,
        start_date,
        end_date,
        bbox,
        save_file_dir,
        **kwargs,
    ):
        """
        Test the get_data method of DataConnector for NASA Earthdata data source.

        This test case verifies if the get_data method of DataConnector correctly
        fetches data from NASA Earthdata for the specified collection, date range,
        bounding box, and bands. It also checks if the returned data array has
        the expected band dimensions.

        Parameters:
            self: The instance of the test class.
            mock_nasa_download_datasets (Mock): A mock object for downloading each band and saving to a raster file. This mock also mocks nasa Earth data api calls and find_items().
            start_date (str): The start date for the data query in 'YYYY-MM-DD' format.
            end_date : The end date for the data query in 'YYYY-MM-DD' format.
            bbox (tuple): The bounding box coordinates (left, bottom, right, top).
            **kwargs: Additional keyword arguments.
        """
        mock_find_items_nasa, mock_get_band, mock_to_raster = (
            mock_nasa_download_datasets
        )

        # Initialize DataConnector
        dc = DataConnector(connector_type=self.connector_type)

        # Call the get_data method
        data_array = dc.connector.get_data(
            data_collection_name=collection,
            date_start=start_date,
            date_end=end_date,
            bands=self.bands,
            bbox=bbox,
        )

        # Assertions
        mock_find_items_nasa.assert_called_once()  # Return 5 dates
        mock_get_band.call_count = len(self.bands) * 5
        mock_to_raster.call_count = 5  # 5 tiles to save
        assert isinstance(data_array, xr.DataArray)
        assert len(data_array.coords["band"]) == len(self.bands)
        assert len(data_array.time) >= 1

    def test_find_data_nasa_earthdata__raises_no_data_found(
        self, mocker, start_date, end_date, bbox
    ):
        mocker.patch(
            "terrakit.download.data_connectors.nasa_earthdata.find_items",
            return_value=[],
        )
        dc = DataConnector(connector_type=self.connector_type)

        with pytest.raises(TerrakitNoDataFoundError):
            dc.connector.find_data(
                data_collection_name="HLSS30_2.0",
                date_start=start_date,
                date_end=end_date,
                bbox=bbox,
                bands=self.bands,
            )
