# © Copyright IBM Corporation 2025-2026
# SPDX-License-Identifier: Apache-2.0


import pytest
import xarray as xr

from unittest.mock import patch
from terrakit import DataConnector
from terrakit.general_utils.exceptions import TerrakitNoDataFoundError
from terrakit.download.data_connectors.nasa_earthdata import get_band


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


class TestGetBandShellInjection:
    """
    get_band previously built a shell command string and called
    subprocess.call(shell=True), allowing shell metacharacters in
    working_dir, band, date, or credential values to execute arbitrary
    commands (remote code execution).

    The fix replaces the string + shell=True call with a plain list
    passed to subprocess.run (no shell=True), so metacharacters are
    treated as literal characters and never interpreted by a shell.
    """

    HREF = (
        "https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com"
        "/sentinel-2-c1-l2a/18/S/UH/2026/9"
        "/S2B_T18SUH_20260901T155021_L2A/B04.tif"
    )
    DATE_ITEMS = [
        {
            "assets": {"B01": {"href": HREF}},
            "properties": {"datetime": "2024-01-06T00:00:00Z"},
        }
    ]
    TEMP_CREDS = {
        "accessKeyId": "FAKE_ACCESS_KEY_ID_FOR_TESTING",
        "secretAccessKey": "FAKE_SECRET_ACCESS_KEY_FOR_TESTING",
        "sessionToken": "token",
    }
    BBOX = [-1.32, 51.06, -1.30, 51.08]

    @pytest.fixture()
    def mock_get_band_io(self, mocker):
        """
        Mock every I/O call inside get_band that is unrelated to subprocess,
        so the test runs without network access or real files.
        """
        mock_crs = mocker.MagicMock()
        mock_crs.to_string.return_value = "EPSG:32630"
        mock_src = mocker.MagicMock()
        mock_src.__enter__ = mocker.MagicMock(return_value=mock_src)
        mock_src.__exit__ = mocker.MagicMock(return_value=False)
        mock_src.crs = mock_crs
        mocker.patch(
            "terrakit.download.data_connectors.nasa_earthdata.rio.open",
            return_value=mock_src,
        )
        mocker.patch(
            "terrakit.download.data_connectors.nasa_earthdata.rio.warp.transform_bounds",
            return_value=(-1.32, 51.06, -1.30, 51.08),
        )
        mock_data = mocker.MagicMock()
        mock_data.rename.return_value = mock_data
        mock_data.expand_dims.return_value = mock_data
        mock_data.max.return_value = mock_data
        mock_data.rio.clip.return_value = mock_data
        mock_data.rio.reproject.return_value = mock_data
        mocker.patch(
            "terrakit.download.data_connectors.nasa_earthdata.rioxarray.open_rasterio",
            return_value=mock_data,
        )

    def _get_subprocess_call(self, mock_subprocess):
        """
        Return (args, kwargs) for whichever of subprocess.call / subprocess.run
        was actually invoked, so assertions work against both the unfixed code
        (uses subprocess.call) and the fixed code (uses subprocess.run).
        """
        if mock_subprocess.run.called:
            return mock_subprocess.run.call_args
        if mock_subprocess.call.called:
            return mock_subprocess.call.call_args
        raise AssertionError(
            "Neither subprocess.run nor subprocess.call was called. "
            "get_band must invoke one of them to build the VRT."
        )

    def test_subprocess_called_with_list_not_shell_string(self, mock_get_band_io):
        """
        subprocess must be invoked with a list of arguments and without
        shell=True.  A list-based call passes arguments directly to execvp,
        so shell metacharacters in any argument are never interpreted.
        """
        with patch(
            "terrakit.download.data_connectors.nasa_earthdata.subprocess"
        ) as mock_sp:
            get_band(self.DATE_ITEMS, "B01", self.BBOX, self.TEMP_CREDS, "/tmp/workdir")
            call_args = self._get_subprocess_call(mock_sp)

        args, kwargs = call_args

        # The first positional argument must be a list, not a string.
        cmd = args[0]
        assert isinstance(cmd, list), (
            "subprocess must receive a list of arguments, not a string. "
            "Passing a string with shell=True enables shell injection."
        )
        assert cmd[0] == "gdalbuildvrt"

        # shell=True must never be set.
        assert kwargs.get("shell") is not True, (
            "shell=True must not be used — it allows shell metacharacters "
            "in any argument to execute arbitrary commands."
        )

    @pytest.mark.parametrize(
        "injection_field,date_items,band,working_dir,temp_creds",
        [
            (
                "working_dir",
                DATE_ITEMS,
                "B01",
                '/tmp/work; echo "injected" > /tmp/pwned',
                TEMP_CREDS,
            ),
            (
                "band",
                [
                    {
                        "assets": {"B01; echo injected": {"href": HREF}},
                        "properties": {"datetime": "2024-01-06T00:00:00Z"},
                    }
                ],
                "B01; echo injected",
                "/tmp/workdir",
                TEMP_CREDS,
            ),
            (
                "accessKeyId",
                DATE_ITEMS,
                "B01",
                "/tmp/workdir",
                {
                    "accessKeyId": "x; touch /tmp/pwned",
                    "secretAccessKey": "secret",
                    "sessionToken": "token",
                },
            ),
        ],
    )
    def test_shell_metacharacters_in_inputs_are_not_executed(
        self,
        mock_get_band_io,
        injection_field,
        date_items,
        band,
        working_dir,
        temp_creds,
    ):
        """
        When shell metacharacters are present in user-controlled inputs,
        subprocess must still be called with a list so those characters
        are passed as literal data to gdalbuildvrt, not to a shell.
        """
        with patch(
            "terrakit.download.data_connectors.nasa_earthdata.subprocess"
        ) as mock_sp:
            get_band(date_items, band, self.BBOX, temp_creds, working_dir)
            call_args = self._get_subprocess_call(mock_sp)

        args, kwargs = call_args
        cmd = args[0]

        assert isinstance(cmd, list), (
            f"Injection via '{injection_field}': subprocess must receive "
            "a list so shell metacharacters are not interpreted."
        )
        assert kwargs.get("shell") is not True, (
            f"Injection via '{injection_field}': shell=True must not be set."
        )
