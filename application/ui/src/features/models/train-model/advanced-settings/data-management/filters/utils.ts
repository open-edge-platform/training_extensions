// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    BoolConfigurableParameter,
    ConfigurableParameterGroup,
    NumberConfigurableParameter,
    TrainingConfiguration,
} from '../../../../../../constants/shared-types';
import {
    findGroupByKey,
    isParameter,
    isParameterGroup,
} from '../../../../model-listing/model-training-parameters/utils';
import { isBoolEnableParameter, isNumberParameter } from '../../utils';

export const getFiltersParameters = (trainingConfiguration: TrainingConfiguration) => {
    const datasetPreparation = findGroupByKey(trainingConfiguration.parameters, 'dataset_preparation')?.parameters;
    return findGroupByKey(datasetPreparation, 'filtering');
};

export type FilterConfigurableParameters = [BoolConfigurableParameter, NumberConfigurableParameter];

export type FilterConfigurableParameterGroup = Omit<ConfigurableParameterGroup, 'parameters'> & {
    parameters: FilterConfigurableParameters;
};

export const isFilterConfigurableParameterGroup = (
    parameters: ConfigurableParameterGroup
): parameters is FilterConfigurableParameterGroup => {
    if (isParameterGroup(parameters)) {
        const filterGroupParameters = parameters.parameters;
        if (filterGroupParameters.length === 2) {
            const [enableParameter, configParameter] = filterGroupParameters;

            return (
                isParameter(enableParameter) &&
                isBoolEnableParameter(enableParameter) &&
                isParameter(configParameter) &&
                isNumberParameter(configParameter)
            );
        }
    }

    return false;
};

export const checkIfFiltersAreEnabled = (parameters: FilterConfigurableParameterGroup[]): boolean => {
    return parameters
        .flatMap((parameterGroup) => parameterGroup.parameters)
        .some((parameter) => isBoolEnableParameter(parameter) && parameter.value === true);
};
