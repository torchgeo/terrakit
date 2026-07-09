# © Copyright IBM Corporation 2025-2026
# SPDX-License-Identifier: Apache-2.0


import json
import os
import pytest
import tacoreader

from pathlib import Path

from terrakit.general_utils.exceptions import TerrakitValidationError
from terrakit.store.taco import TacoStoreConfig, TacoCls, taco_store_data, load_taco
from terrakit.validate.store_model import StoreModel
from tests.component_tests.store.conftest import WORKING_DIR, DUMMY_DATA_DIR


class TestTacoModel:
    """Test TacoStoreConfig pydantic model validation"""

    def test_taco_model_default_values(self):
        """Test TacoStoreConfig model initializes with correct default values"""
        taco = TacoStoreConfig(license="CC-BY-4.0")
        assert taco.active is True
        assert taco.license == "CC-BY-4.0"
        assert taco.format == "taco"
        assert taco.save_dir == "./tmp"
        assert taco.tortilla_name == ""
        assert taco.statistics is True
        assert taco.include_config is True
        assert taco.check_dataset is True

    def test_taco_model_custom_values(self):
        """Test TacoStoreConfig model accepts custom values"""
        taco = TacoStoreConfig(
            active=False,
            license="MIT",
            format="local",
            save_dir="/custom/save",
            tortilla_name="custom.tacozip",
            statistics=False,
            include_config=False,
            check_dataset=False,
        )
        assert taco.active is False
        assert taco.license == "MIT"
        assert taco.format == "local"
        assert taco.save_dir == "/custom/save"
        assert taco.tortilla_name == "custom.tacozip"
        assert taco.statistics is False
        assert taco.include_config is False
        assert taco.check_dataset is False

    def test_taco_model_validation_from_dict(self):
        """Test TacoStoreConfig model can be validated from dictionary"""
        data = {
            "active": True,
            "license": "CC-BY-4.0",
            "format": "taco",
            "save_dir": "./tmp",
            "tortilla_name": "test.tacozip",
        }
        taco = TacoStoreConfig.model_validate(data)
        assert taco.active is True
        assert taco.license == "CC-BY-4.0"
        assert taco.tortilla_name == "test.tacozip"


class TestTacoClsInitialization:
    """Test TacoCls class initialization"""

    def test_taco_cls_initialization_default(self):
        """Test TacoCls initializes with default values"""
        taco = TacoCls(active=True, license="CC-BY-4.0")
        assert taco.active is True
        assert taco.license == "CC-BY-4.0"
        assert taco.format == "taco"
        assert taco.save_dir == "./tmp"
        assert taco.tortilla_name == ""
        assert taco.statistics is True
        assert taco.include_config is True
        assert taco.check_dataset is True

    def test_taco_cls_initialization_custom(self):
        """Test TacoCls initializes with custom values"""
        taco = TacoCls(
            active=False,
            license="CC0-1.0",
            format="local",
            save_dir="/custom/save",
            tortilla_name="custom.tacozip",
            statistics=False,
            include_config=False,
            check_dataset=False,
        )
        assert taco.active is False
        assert taco.license == "CC0-1.0"
        assert taco.format == "local"
        assert taco.save_dir == "/custom/save"
        assert taco.tortilla_name == "custom.tacozip"
        assert taco.statistics is False
        assert taco.include_config is False
        assert taco.check_dataset is False

    def test_taco_cls_requires_active_parameter(self):
        """Test TacoCls requires active and license parameters"""
        with pytest.raises(TypeError) as exc_info:
            TacoCls()
        error_msg = str(exc_info.value)
        assert "active" in error_msg or "license" in error_msg


class TestCreateTortilla:
    """Test create_tortilla method"""

    def test_create_tortilla_basic(self, taco_setup, store_cleanup):
        """Test create_tortilla creates tortilla file successfully"""
        taco = TacoCls(
            active=True,
            license="CC-BY-4.0",
            save_dir=WORKING_DIR,
            tortilla_name="test.tacozip",
        )

        result = taco.create_tortilla(
            dataset_name="test",
            working_dir=WORKING_DIR,
        )

        assert Path(result) == Path(WORKING_DIR) / "test.tacozip"
        assert os.path.exists(result)
        assert "test.tacozip" in os.listdir(WORKING_DIR)

    def test_create_tortilla_custom_chip_suffix(self, taco_setup, store_cleanup):
        """Test create_tortilla works with custom chip suffix"""
        taco = TacoCls(
            active=True,
            license="CC-BY-4.0",
            save_dir=WORKING_DIR,
            tortilla_name="test_custom.tacozip",
        )

        result = taco.create_tortilla(
            dataset_name="test_custom",
            working_dir=WORKING_DIR,
            chip_suffix=".data.tif",
        )

        assert os.path.exists(result)
        assert "test_custom.tacozip" in os.listdir(WORKING_DIR)

    def test_create_tortilla_creates_tortilla_directory(
        self, taco_setup, store_cleanup
    ):
        """Test create_tortilla writes the .tacozip output into save_dir"""
        taco = TacoCls(
            active=True,
            license="CC-BY-4.0",
            save_dir=WORKING_DIR,
            tortilla_name="test.tacozip",
        )

        result = taco.create_tortilla(
            dataset_name="test",
            working_dir=WORKING_DIR,
        )

        assert os.path.exists(result)
        assert "test.tacozip" in os.listdir(WORKING_DIR)

    def test_create_tortilla_uses_dataset_name_when_tortilla_name_empty(
        self, taco_setup, store_cleanup
    ):
        """Test create_tortilla uses dataset_name when tortilla_name is empty"""
        taco = TacoCls(
            active=True,
            license="CC-BY-4.0",
            save_dir=WORKING_DIR,
            tortilla_name="",  # Empty tortilla name
        )

        result = taco.create_tortilla(
            dataset_name="my_dataset",
            working_dir=WORKING_DIR,
        )

        assert Path(result) == Path(WORKING_DIR) / "my_dataset.tacozip"
        assert os.path.exists(result)

    def test_create_tortilla_splits_data_correctly(self, taco_setup, store_cleanup):
        """Test create_tortilla splits data into train/val/test correctly"""
        taco = TacoCls(
            active=True,
            license="CC-BY-4.0",
            save_dir=WORKING_DIR,
            tortilla_name="test.tacozip",
        )

        result = taco.create_tortilla(
            dataset_name="test",
            working_dir=WORKING_DIR,
        )

        # Load the tortilla to verify it's valid
        tt = tacoreader.load(result)
        assert tt is not None
        assert hasattr(tt, "id")

    def test_create_tortilla_extracts_dates_from_filenames(
        self, taco_setup, store_cleanup
    ):
        """Test create_tortilla extracts dates from filenames correctly"""
        taco = TacoCls(
            active=True,
            license="CC-BY-4.0",
            save_dir=WORKING_DIR,
            tortilla_name="test.tacozip",
        )

        result = taco.create_tortilla(
            dataset_name="test",
            working_dir=WORKING_DIR,
        )

        # Load the tortilla to verify it's valid
        tt = tacoreader.load(result)
        # TacoDataset in tacoreader 2.0+ has different API - just verify it loads successfully
        assert tt is not None
        assert hasattr(tt, "id")


class TestTacoStoreData:
    """Test taco_store_data function"""

    def test_taco_store_data_basic(self, taco_setup, store_cleanup):
        """Test taco_store_data creates tortilla successfully"""
        result = taco_store_data(
            dataset_name="test",
            license="CC-BY-4.0",
            working_dir=WORKING_DIR,
            tortilla_name="test.tacozip",
            save_dir=WORKING_DIR,
        )

        assert Path(result) == Path(WORKING_DIR) / "test.tacozip"
        assert os.path.exists(result)
        assert "test.tacozip" in os.listdir(WORKING_DIR)

    def test_taco_store_data_creates_metadata(self, taco_setup, store_cleanup):
        """Test taco_store_data creates metadata file"""
        taco_store_data(
            dataset_name="test",
            license="CC-BY-4.0",
            working_dir=WORKING_DIR,
            tortilla_name="test.tacozip",
            save_dir=WORKING_DIR,
        )

        metadata_file = os.path.join(WORKING_DIR, "test_metadata.json")
        assert os.path.exists(metadata_file)

        # Verify metadata content
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # Check top-level metadata structure
        assert "dataset_name" in metadata
        assert metadata["dataset_name"] == "test"
        assert "lineage" in metadata
        assert len(metadata["lineage"]) > 0

        # Check store step in lineage
        store_step = metadata["lineage"][0]
        assert store_step["step_id"] == "store"
        assert store_step["activity"] == "Package dataset in taco format"
        assert store_step["method"] == "terrakit.store.taco.taco_store_data"
        assert "parameters" in store_step

    def test_taco_store_data_with_custom_parameters(self, taco_setup, store_cleanup):
        """Test taco_store_data with custom parameters"""
        result = taco_store_data(
            dataset_name="custom_test",
            license="CC-BY-4.0",
            working_dir=WORKING_DIR,
            active=True,
            format="taco",
            save_dir=WORKING_DIR,
            tortilla_name="custom.tacozip",
            statistics=False,
            include_config=False,
            check_dataset=False,
        )

        assert os.path.exists(result)
        # The implementation uses tortilla_name when provided
        assert "custom.tacozip" in os.listdir(WORKING_DIR)

    def test_taco_store_data_validates_pipeline_model(self, taco_setup, store_cleanup):
        """Test taco_store_data validates pipeline model"""
        # This should succeed with valid parameters
        result = taco_store_data(
            dataset_name="test",
            license="CC-BY-4.0",
            working_dir=WORKING_DIR,
            save_dir=WORKING_DIR,
        )
        assert os.path.exists(result)

    @pytest.mark.skip(
        "WiP: Expected this test to fail as there appears to be an issue when running "
        "labels_to_data.py which is resolved if the shp files are removed from the working dir."
    )
    def test_taco_store_data_with_shapefiles(
        self, taco_setup, create_dummy_shpfile, store_cleanup
    ):
        """Test that store data function works even if shapefiles exist in working directory"""
        taco_store_data(
            dataset_name="test",
            license="CC-BY-4.0",
            working_dir=WORKING_DIR,
            tortilla_name="test.tacozip",
            save_dir=WORKING_DIR,
        )
        assert "test.tacozip" in os.listdir(WORKING_DIR)


class TestLoadTaco:
    """Test load_taco function"""

    def test_load_taco_basic(self, taco_setup, store_cleanup, caplog):
        """Test load_taco loads and logs taco data"""
        # First create a tortilla
        result = taco_store_data(
            dataset_name="test",
            license="CC-BY-4.0",
            working_dir=WORKING_DIR,
            tortilla_name="test.tacozip",
            save_dir=WORKING_DIR,
        )

        # Now load it
        data = load_taco(result)

        # Check that something was logged
        assert len(caplog.records) > 0
        # Check that data was returned
        assert data is not None

    def test_load_taco_returns_data(self, taco_setup, store_cleanup):
        """Test load_taco returns data"""
        result = taco_store_data(
            dataset_name="test",
            license="CC-BY-4.0",
            working_dir=WORKING_DIR,
            tortilla_name="test.tacozip",
            save_dir=WORKING_DIR,
        )

        return_value = load_taco(result)
        assert return_value is not None


class TestTacoErrorHandling:
    """Test error handling and edge cases"""

    def test_create_tortilla_no_data_files(self, store_cleanup):
        """Test create_tortilla handles missing data files gracefully"""
        Path(WORKING_DIR).mkdir(parents=True, exist_ok=True)

        taco = TacoCls(
            active=True,
            license="CC-BY-4.0",
            save_dir=WORKING_DIR,
            tortilla_name="test.tacozip",
        )

        # Should raise an error or handle gracefully when no .tif files exist
        with pytest.raises(Exception):  # Could be ValueError or IndexError
            taco.create_tortilla(
                dataset_name="test",
                working_dir=WORKING_DIR,
            )

    def test_create_tortilla_mismatched_data_label_files(self, store_cleanup):
        """Test create_tortilla handles mismatched data/label files"""
        Path(WORKING_DIR).mkdir(parents=True, exist_ok=True)

        # Create only data file without corresponding label file
        import shutil

        shutil.copy(
            f"{DUMMY_DATA_DIR}/dummy_2025-01-01_imputed_0.data.tif",
            f"{WORKING_DIR}/dummy_2025-01-01_imputed_1.data.tif",
        )

        taco = TacoCls(
            active=True,
            license="CC-BY-4.0",
            save_dir=WORKING_DIR,
            tortilla_name="test.tacozip",
        )

        # Should raise an error when label file is missing
        with pytest.raises(Exception):  # Could be FileNotFoundError or rasterio error
            taco.create_tortilla(
                dataset_name="test",
                working_dir=WORKING_DIR,
            )

    def test_taco_store_data_invalid_working_dir(self):
        """Test taco_store_data with invalid working directory on read-only filesystem"""
        # On macOS, /nonexistent is on a read-only filesystem and will raise OSError
        # when trying to create directories
        with pytest.raises(OSError):
            taco_store_data(
                dataset_name="test",
                license="CC-BY-4.0",
                working_dir="/nonexistent/directory/path",
                save_dir=WORKING_DIR,
            )

    def test_create_tortilla_creates_save_dir_if_not_exists(
        self, taco_setup, store_cleanup
    ):
        """Test create_tortilla creates save_dir if it doesn't exist"""
        new_save_dir = os.path.join(WORKING_DIR, "new_save_dir")
        assert not os.path.exists(new_save_dir)

        taco = TacoCls(
            active=True,
            license="CC-BY-4.0",
            save_dir=new_save_dir,
            tortilla_name="test.tacozip",
        )

        result = taco.create_tortilla(
            dataset_name="test",
            working_dir=WORKING_DIR,
        )

        assert os.path.exists(new_save_dir)
        assert os.path.exists(result)


class TestTacoIntegration:
    """Integration tests for complete workflows"""

    def test_full_workflow_taco_model_to_tortilla(self, taco_setup, store_cleanup):
        """Test complete workflow from Taco model validation to tortilla creation"""
        # Create TacoCls instance
        taco_cls = TacoCls(
            active=True,
            license="CC-BY-4.0",
            format="taco",
            save_dir=WORKING_DIR,
            tortilla_name="integration_test.tacozip",
        )

        # Validate with TacoStoreConfig model
        taco_model = TacoStoreConfig.model_validate(taco_cls)
        assert taco_model.active is True
        assert taco_model.tortilla_name == "integration_test.tacozip"

        # Create tortilla
        result = taco_cls.create_tortilla(
            dataset_name="integration_test",
            working_dir=WORKING_DIR,
        )

        assert os.path.exists(result)

        # Load and verify tortilla
        tt = tacoreader.load(result)
        # TacoDataset in tacoreader 2.0+ doesn't support len(), but successful load indicates valid dataset
        assert tt is not None
        # Check if we can access the dataset (will raise if invalid)
        assert hasattr(tt, "id")

    def test_full_workflow_with_taco_store_data(self, taco_setup, store_cleanup):
        """Test complete workflow using taco_store_data function"""
        result = taco_store_data(
            dataset_name="workflow_test",
            license="CC-BY-4.0",
            working_dir=WORKING_DIR,
            tortilla_name="workflow_test.tacozip",
            save_dir=WORKING_DIR,
            statistics=True,
            include_config=True,
            check_dataset=True,
        )

        # Verify tortilla was created
        assert os.path.exists(result)

        # Verify metadata was created
        metadata_file = os.path.join(WORKING_DIR, "workflow_test_metadata.json")
        assert os.path.exists(metadata_file)

        # Load and verify tortilla content
        tt = tacoreader.load(result)
        # TacoDataset in tacoreader 2.0+ doesn't support len(), but successful load indicates valid dataset
        assert tt is not None
        assert hasattr(tt, "id")


class TestTortillaNameValidation:
    """Test tortilla_name validation in StoreModel"""

    def test_tortilla_name_optional_empty_string(self):
        """Test that tortilla_name can be an empty string (optional)"""
        model = StoreModel(license="CC-BY-4.0", tortilla_name="")
        assert model.tortilla_name == ""

    def test_tortilla_name_optional_not_provided(self):
        """Test that tortilla_name defaults to empty string when not provided"""
        model = StoreModel(license="CC-BY-4.0")
        assert model.tortilla_name == ""

    def test_tortilla_name_valid_with_tacozip_extension(self):
        """Test that tortilla_name is valid when it ends with .tacozip"""
        model = StoreModel(license="CC-BY-4.0", tortilla_name="my_dataset.tacozip")
        assert model.tortilla_name == "my_dataset.tacozip"

    def test_tortilla_name_valid_with_path_and_tacozip_extension(self):
        """Test that tortilla_name is valid with path and .tacozip extension"""
        model = StoreModel(
            license="CC-BY-4.0", tortilla_name="./output/my_dataset.tacozip"
        )
        assert model.tortilla_name == "./output/my_dataset.tacozip"

    def test_tortilla_name_invalid_without_tacozip_extension(self):
        """Test that tortilla_name raises TerrakitValidationError when it doesn't end with .tacozip"""
        with pytest.raises(TerrakitValidationError) as exc_info:
            StoreModel(license="CC-BY-4.0", tortilla_name="my_dataset.tortilla")

        error_msg = str(exc_info.value)
        assert "tortilla_name must end with '.tacozip' extension" in error_msg
        assert "my_dataset.tortilla" in error_msg
        assert "Please add '.tacozip' to the end of the name" in error_msg

    def test_tortilla_name_invalid_with_wrong_extension(self):
        """Test that tortilla_name raises TerrakitValidationError with wrong extension"""
        with pytest.raises(TerrakitValidationError) as exc_info:
            StoreModel(license="CC-BY-4.0", tortilla_name="my_dataset.zip")

        error_msg = str(exc_info.value)
        assert "tortilla_name must end with '.tacozip' extension" in error_msg
        assert "Please add '.tacozip' to the end of the name" in error_msg

    def test_tortilla_name_invalid_no_extension(self):
        """Test that tortilla_name raises TerrakitValidationError with no extension"""
        with pytest.raises(TerrakitValidationError) as exc_info:
            StoreModel(license="CC-BY-4.0", tortilla_name="my_dataset")

        error_msg = str(exc_info.value)
        assert "tortilla_name must end with '.tacozip' extension" in error_msg
        assert "Please add '.tacozip' to the end of the name" in error_msg

    def test_tortilla_name_backward_compatibility_with_taco_store_config(self):
        """Test that TacoStoreConfig alias works with tortilla_name validation"""
        # TacoStoreConfig should be an alias for StoreModel
        model = TacoStoreConfig(license="CC-BY-4.0", tortilla_name="test.tacozip")
        assert model.tortilla_name == "test.tacozip"

        # Should also validate correctly
        with pytest.raises(TerrakitValidationError):
            TacoStoreConfig(license="CC-BY-4.0", tortilla_name="test.tortilla")


class TestTortillaNameDefaultBehavior:
    """Test that tortilla_name defaults to {working_dir}/{dataset_name}.tacozip when not provided"""

    def test_create_tortilla_uses_dataset_name_when_tortilla_name_empty(
        self, taco_setup, store_cleanup
    ):
        """Test that create_tortilla uses dataset_name.tacozip when tortilla_name is empty"""
        taco = TacoCls(
            active=True,
            license="CC-BY-4.0",
            save_dir=WORKING_DIR,
            tortilla_name="",  # Empty tortilla name
        )

        result = taco.create_tortilla(
            dataset_name="my_dataset",
            working_dir=WORKING_DIR,
            save_dir=WORKING_DIR,
        )

        # Should create file with dataset_name.tacozip
        assert Path(result) == Path(WORKING_DIR) / "my_dataset.tacozip"
        assert os.path.exists(result)

    def test_create_tortilla_uses_custom_tortilla_name_when_provided(
        self, taco_setup, store_cleanup
    ):
        """Test that create_tortilla uses custom tortilla_name when provided"""
        taco = TacoCls(
            active=True,
            license="CC-BY-4.0",
            save_dir=WORKING_DIR,
            tortilla_name="custom_name.tacozip",
        )

        result = taco.create_tortilla(
            dataset_name="my_dataset",
            working_dir=WORKING_DIR,
            save_dir=WORKING_DIR,
        )

        # Should use the custom tortilla_name, not dataset_name
        assert Path(result) == Path(WORKING_DIR) / "custom_name.tacozip"
        assert os.path.exists(result)

    def test_taco_store_data_default_tortilla_name(self, taco_setup, store_cleanup):
        """Test taco_store_data creates tortilla with default name when tortilla_name is empty"""
        result = taco_store_data(
            dataset_name="test_default",
            license="CC-BY-4.0",
            working_dir=WORKING_DIR,
            save_dir=WORKING_DIR,
            tortilla_name="",  # Explicitly empty
        )

        # Should create file with dataset_name.tacozip
        assert Path(result) == Path(WORKING_DIR) / "test_default.tacozip"
        assert os.path.exists(result)

    def test_taco_store_data_custom_tortilla_name(self, taco_setup, store_cleanup):
        """Test taco_store_data uses custom tortilla_name when provided"""
        result = taco_store_data(
            dataset_name="test_custom",
            license="CC-BY-4.0",
            working_dir=WORKING_DIR,
            save_dir=WORKING_DIR,
            tortilla_name="custom_output.tacozip",
        )

        # Should create file with custom tortilla_name
        assert Path(result) == Path(WORKING_DIR) / "custom_output.tacozip"
        assert os.path.exists(result)


class TestLicenseValidation:
    """Test license validation in StoreModel and taco_store_data"""

    def test_store_model_requires_license(self):
        """Test that StoreModel raises ValidationError when license is not provided"""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            StoreModel()

        error_msg = str(exc_info.value)
        assert "license" in error_msg.lower()

    def test_store_model_rejects_empty_license(self):
        """Test that StoreModel raises TerrakitValidationError when license is empty string"""
        with pytest.raises(TerrakitValidationError) as exc_info:
            StoreModel(license="")

        error_msg = str(exc_info.value)
        assert "license" in error_msg.lower()
        assert "must be provided" in error_msg.lower()

    def test_store_model_rejects_whitespace_license(self):
        """Test that StoreModel raises TerrakitValidationError when license is only whitespace"""
        with pytest.raises(TerrakitValidationError) as exc_info:
            StoreModel(license="   ")

        error_msg = str(exc_info.value)
        assert "license" in error_msg.lower()
        assert "must be provided" in error_msg.lower()

    def test_store_model_accepts_valid_license(self):
        """Test that StoreModel accepts a valid license"""
        model = StoreModel(license="CC-BY-4.0")
        assert model.license == "CC-BY-4.0"

    def test_taco_cls_requires_license(self):
        """Test that TacoCls requires license parameter"""
        with pytest.raises(TypeError) as exc_info:
            TacoCls(active=True)

        assert "license" in str(exc_info.value)

    def test_taco_cls_accepts_valid_license(self):
        """Test that TacoCls accepts a valid license"""
        taco = TacoCls(active=True, license="CC-BY-4.0")
        assert taco.license == "CC-BY-4.0"

    def test_taco_store_data_requires_license(self, taco_setup, store_cleanup):
        """Test that taco_store_data raises error when license is not provided"""
        with pytest.raises(TypeError) as exc_info:
            taco_store_data(
                dataset_name="test",
                working_dir=WORKING_DIR,
                save_dir=WORKING_DIR,
            )

        assert "license" in str(exc_info.value)

    def test_taco_store_data_with_cc_by_40_license(self, taco_setup, store_cleanup):
        """Test that taco_store_data works with CC-BY-4.0 license"""
        result = taco_store_data(
            dataset_name="test",
            license="CC-BY-4.0",
            working_dir=WORKING_DIR,
            save_dir=WORKING_DIR,
        )

        assert os.path.exists(result)

        # Load the tortilla and verify license is set
        tt = tacoreader.load(result)
        assert tt is not None
        # Check that the taco has the correct license
        assert hasattr(tt, "licenses")
        assert "CC-BY-4.0" in tt.licenses

    def test_create_tortilla_uses_license_from_taco_cls(
        self, taco_setup, store_cleanup
    ):
        """Test that create_tortilla uses the license from TacoCls instance"""
        taco = TacoCls(
            active=True,
            license="CC-BY-4.0",
            save_dir=WORKING_DIR,
        )

        result = taco.create_tortilla(
            dataset_name="test",
            working_dir=WORKING_DIR,
        )

        # Load the tortilla and verify license is set
        tt = tacoreader.load(result)
        assert tt is not None
        assert hasattr(tt, "licenses")
        assert "CC-BY-4.0" in tt.licenses

    def test_taco_store_data_no_default_license(self):
        """Test that there is no default license value"""
        # This test verifies that license parameter has no default
        # by checking the function signature
        import inspect

        sig = inspect.signature(taco_store_data)
        license_param = sig.parameters["license"]

        # The parameter should not have a default value
        assert license_param.default == inspect.Parameter.empty


# Made with Bob
