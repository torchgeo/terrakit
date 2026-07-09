# © Copyright IBM Corporation 2026
# SPDX-License-Identifier: Apache-2.0


import logging

from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    ValidationInfo,
)

from terrakit.general_utils.exceptions import TerrakitValidationError


logger = logging.getLogger(__name__)


class StoreModel(BaseModel):
    """
    Model for configuration of the store TerraKit pipeline step.

    Attributes:
        model_config (ConfigDict): Configuration dictionary for the model.
        active (bool): Indicates if the store step is active. Default is True.
        format (str): Format for storing data. Default is "taco".
        save_dir (str): Directory to save the tortilla files. Default is "./tmp".
        tortilla_name (str): Name of the final tortilla file. Optional - if not provided, defaults to {dataset_name}.tacozip.
        license (str): License name for the dataset. Required - no default value.
        statistics (bool): Whether to include statistics. Default is True.
        include_config (bool): Whether to include configuration. Default is True.
        check_dataset (bool): Whether to check the dataset. Default is True.
    """

    model_config = ConfigDict(from_attributes=True)

    active: bool = True
    format: str = "taco"
    save_dir: str = "./tmp"
    tortilla_name: str = ""
    license: str
    statistics: bool = True
    include_config: bool = True
    check_dataset: bool = True

    @field_validator("license", mode="after")
    def validate_license(cls, v: str, info: ValidationInfo) -> str:
        """
        Validates that a license name is provided.

        Args:
            v: The license value to validate
            info: Validation context information

        Returns:
            The validated license

        Raises:
            TerrakitValidationError: If license is not provided or is empty
        """
        if not v or v.strip() == "":
            raise TerrakitValidationError(
                "A license name must be provided for the dataset. "
                "Please specify a license (e.g., 'CC-BY-4.0')."
            )

        return v

    @field_validator("tortilla_name", mode="after")
    def validate_tortilla_name(cls, v: str, _info: ValidationInfo) -> str:
        """
        Validates that tortilla_name ends with .tacozip if provided.

        The tortilla_name is optional:
        - If empty string, it will be set to default {dataset_name}.tacozip during execution
        - If provided, it must end with .tacozip extension

        Args:
            v: The tortilla_name value to validate
            info: Validation context information

        Returns:
            The validated tortilla_name

        Raises:
            TerrakitValidationError: If tortilla_name is provided but doesn't end with .tacozip
        """
        # Empty string is valid - will use default
        if v == "":
            return v

        # If provided, must end with .tacozip
        if not v.endswith(".tacozip"):
            raise TerrakitValidationError(
                f"tortilla_name must end with '.tacozip' extension. Got: '{v}'. "
                f"Please add '.tacozip' to the end of the name."
            )

        return v
