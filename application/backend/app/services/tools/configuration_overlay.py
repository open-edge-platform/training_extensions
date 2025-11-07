# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

from app.models.training_configuration.configuration import PartialTrainingConfiguration, TrainingConfiguration


class ConfigurationOverlayTools:
    @classmethod
    def delete_none_from_dict(cls, d: dict) -> dict:
        """
        Remove None values recursively from dictionaries.

        Args:
            d (dict): Dictionary to process.

        Returns:
            dict: Dictionary with None values removed.
        """
        for key, value in list(d.items()):
            if isinstance(value, dict):
                cls.delete_none_from_dict(value)
                if not value:
                    del d[key]
            elif value is None:
                del d[key]
            elif isinstance(value, list):
                for i, v_i in enumerate(value):
                    if isinstance(v_i, dict):
                        cls.delete_none_from_dict(v_i)
                        if not v_i:
                            del value[i]
                if not value:
                    del d[key]
        return d

    @classmethod
    def merge_deep_dict(cls, a: dict, b: dict, common_parameters_only: bool = False) -> dict:
        """
        Recursively merge dictionaries 'b' into 'a' with deep dictionary support.

        This method merges keys and values from dictionary 'b' into dictionary 'a'.
        For nested dictionaries, it performs a recursive merge. For all other value types,
        values from 'b' overwrite values in 'a' when keys exist in both dictionaries.

        Example:
            a = {'x': 1, 'y': {'a': 2}}
            b = {'y': {'b': 3}, 'z': 4}
            result = {'x': 1, 'y': {'a': 2, 'b': 3}, 'z': 4}

        Args:
            a (dict): Target dictionary to merge into (modified in-place).
            b (dict): Source dictionary whose values will be merged into 'a'.
            common_parameters_only (bool): If True, only parameters present in both dictionaries
                will be updated. Defaults to False.

        Returns:
            dict: The modified dictionary 'a' containing merged values from 'b'.
        """
        for key, val in b.items():
            if key in a:
                if isinstance(a[key], dict) and isinstance(val, dict):
                    cls.merge_deep_dict(a[key], val, common_parameters_only)
                else:
                    a[key] = val
            elif not common_parameters_only:
                a[key] = val
        return a

    @classmethod
    def overlay_training_configurations(
        cls,
        base_config: PartialTrainingConfiguration,
        *overlaying_configs: PartialTrainingConfiguration,
        validate_full_config: bool = True,
        common_hyperparameters_only: bool = False,
    ) -> TrainingConfiguration | PartialTrainingConfiguration:
        """
        Overlays multiple training configurations on top of a base configuration.

        This method takes a base configuration and applies successive overlay configurations
        on top of it, merging dictionaries deeply. The result can be validated as either
        a full or partial training configuration.

        Args:
            base_config (PartialTrainingConfiguration): The base configuration to start with.
            *overlaying_configs (PartialTrainingConfiguration): Variable number of configurations
                to overlay on the base.
            validate_full_config (bool): If True, validates result as a full TrainingConfiguration,
                otherwise as a PartialTrainingConfiguration. Defaults to True.
            common_hyperparameters_only (bool): If True, only hyperparameters present in both the base
                and overlaying configurations will be updated; parameters unique to the overlaying
                configs will be ignored. If False, all parameters from overlaying configs are merged
                into the base. Defaults to False.

        Returns:
            TrainingConfiguration | PartialTrainingConfiguration: The merged configuration, either as
                TrainingConfiguration or PartialTrainingConfiguration.
        """
        base_config_dict = cls.delete_none_from_dict(base_config.model_dump())

        overlay_config_dict = deepcopy(base_config_dict)
        for config in overlaying_configs:
            config_dict = cls.delete_none_from_dict(config.model_dump())
            overlay_config_dict = cls.merge_deep_dict(overlay_config_dict, config_dict)

        # overlay hyperparameters separately to ensure that no extra hyperparameters are added
        if common_hyperparameters_only and "hyperparameters" in base_config_dict:
            overlay_config_dict["hyperparameters"] = cls.merge_deep_dict(
                base_config_dict["hyperparameters"],
                overlay_config_dict["hyperparameters"],
                common_parameters_only=True,
            )

        if validate_full_config:
            return TrainingConfiguration.model_validate(overlay_config_dict)
        return PartialTrainingConfiguration.model_validate(overlay_config_dict)
