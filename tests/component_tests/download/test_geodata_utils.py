# © Copyright IBM Corporation 2026
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
import rasterio
import rasterio.errors
import tempfile
import xarray as xr
from pathlib import Path

from terrakit.download.geodata_utils import save_cog
from terrakit.download.geodata_utils import check_projection


class TestSaveCog:
    """Test suite for save_cog function to ensure band names are preserved correctly."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def multi_band_dataarray(self):
        """Create a multi-band xarray DataArray with named bands."""
        # Create a simple multi-band DataArray
        data = np.random.rand(3, 10, 10)  # 3 bands, 10x10 pixels
        coords = {
            "band": ["temperature", "precipitation", "wind_speed"],
            "y": np.linspace(40, 41, 10),
            "x": np.linspace(-91, -90, 10),
        }
        da = xr.DataArray(
            data,
            coords=coords,
            dims=["band", "y", "x"],
            attrs={"crs": "EPSG:4326"},
        )
        # Set spatial dimensions and CRS
        da = da.rio.write_crs("EPSG:4326")
        da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
        return da

    @pytest.fixture
    def single_band_dataarray(self):
        """Create a single-band xarray DataArray."""
        data = np.random.rand(1, 10, 10)  # 1 band, 10x10 pixels
        coords = {
            "band": ["temperature"],
            "y": np.linspace(40, 41, 10),
            "x": np.linspace(-91, -90, 10),
        }
        da = xr.DataArray(
            data,
            coords=coords,
            dims=["band", "y", "x"],
            attrs={"crs": "EPSG:4326"},
        )
        da = da.rio.write_crs("EPSG:4326")
        da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
        return da

    def test_save_cog_preserves_multi_band_names(self, temp_dir, multi_band_dataarray):
        """Test that save_cog correctly preserves band names for multi-band GeoTIFF."""
        output_file = temp_dir / "test_multi_band.tif"

        # Save the DataArray using save_cog (it expects a Dataset)
        save_cog(
            ds=multi_band_dataarray,
            filename=str(output_file),
        )

        # Verify file was created
        assert output_file.exists(), "Output file was not created"

        # Open the saved file and check band descriptions
        with rasterio.open(output_file) as src:
            assert src.count == 3, f"Expected 3 bands, got {src.count}"

            # Check each band description
            band_descriptions = [src.descriptions[i] for i in range(src.count)]
            expected_names = ["temperature", "precipitation", "wind_speed"]

            assert band_descriptions == expected_names, (
                f"Band descriptions do not match. "
                f"Expected: {expected_names}, Got: {band_descriptions}"
            )

    def test_save_cog_preserves_single_band_name(self, temp_dir, single_band_dataarray):
        """Test that save_cog correctly preserves band name for single-band GeoTIFF."""
        output_file = temp_dir / "test_single_band.tif"

        # Save the DataArray using save_cog
        save_cog(
            ds=single_band_dataarray,
            filename=str(output_file),
        )

        # Verify file was created
        assert output_file.exists(), "Output file was not created"

        # Open the saved file and check band description
        with rasterio.open(output_file) as src:
            assert src.count == 1, f"Expected 1 band, got {src.count}"

            band_description = src.descriptions[0]
            expected_name = "temperature"

            assert band_description == expected_name, (
                f"Band description does not match. "
                f"Expected: {expected_name}, Got: {band_description}"
            )

    def test_save_cog_with_special_characters_in_names(self, temp_dir):
        """Test that save_cog handles band names with special characters (e.g., CDS variables)."""
        output_file = temp_dir / "test_special_chars.tif"

        # Create DataArray with special character names like CDS returns
        data = np.random.rand(3, 10, 10)
        coords = {
            "band": ["2m_temperature", "mean_total_precipitation_rate", "10m_wind"],
            "y": np.linspace(40, 41, 10),
            "x": np.linspace(-91, -90, 10),
        }
        da = xr.DataArray(data, coords=coords, dims=["band", "y", "x"])
        da = da.rio.write_crs("EPSG:4326")
        da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")

        save_cog(ds=da, filename=str(output_file))

        assert output_file.exists(), "Output file was not created"

        with rasterio.open(output_file) as src:
            band_descriptions = [src.descriptions[i] for i in range(src.count)]
            expected_names = [
                "2m_temperature",
                "mean_total_precipitation_rate",
                "10m_wind",
            ]
            assert band_descriptions == expected_names, (
                f"Band descriptions do not match. "
                f"Expected: {expected_names}, Got: {band_descriptions}"
            )

    def test_save_cog_readable_in_qgis(self, temp_dir, multi_band_dataarray):
        """
        Test that saved GeoTIFF has proper metadata that would be readable in QGIS.
        This validates the fix for the original issue where QGIS showed all bands
        with the same name.
        """
        output_file = temp_dir / "test_qgis_readable.tif"

        save_cog(
            ds=multi_band_dataarray,
            filename=str(output_file),
        )

        # Verify all metadata that QGIS would read
        with rasterio.open(output_file) as src:
            # Check band count
            assert src.count == 3, "Incorrect band count"

            # Check CRS
            assert src.crs is not None, "CRS not set"

            # Check that each band has a unique description
            descriptions = [src.descriptions[i] for i in range(src.count)]
            assert len(set(descriptions)) == len(descriptions), (
                "Band descriptions are not unique - QGIS would show duplicate names"
            )

            # Verify descriptions match expected names from the fixture
            expected_names = ["temperature", "precipitation", "wind_speed"]
            assert descriptions == expected_names, (
                f"Band descriptions don't match. Expected: {expected_names}, Got: {descriptions}"
            )


def test_check_projection_command_injection(tmp_path):
    """Ensure malicious inputs cannot execute arbitrary shell commands."""
    canary_file = tmp_path / "pwned.txt"

    # Payload designed to create the canary file if interpreted by a shell
    malicious_input = f"--help; touch {canary_file}; echo 3"

    # The function should fail safely without running the payload
    with pytest.raises((rasterio.errors.RasterioIOError, FileNotFoundError, Exception)):
        check_projection(malicious_input)

    # Verify command was not executed
    assert not canary_file.exists(), (
        "Command injection vulnerability detected: canary file was created!"
    )


@pytest.mark.parametrize(
    "payload",
    [
        "file.tif; touch {canary}",
        "file.tif && touch {canary}",
        "file.tif | touch {canary}",
        "`touch {canary}`",
        "$(touch {canary})",
    ],
)
def test_check_projection_injection_variants(tmp_path, payload):
    """Test various shell metacharacter injection payloads."""
    canary_file = tmp_path / "pwned_variant.txt"
    formatted_payload = payload.format(canary=canary_file)

    with pytest.raises(Exception):
        check_projection(formatted_payload)

    assert not canary_file.exists(), (
        f"Payload '{formatted_payload}' executed arbitrary command!"
    )


# Made with Bob
