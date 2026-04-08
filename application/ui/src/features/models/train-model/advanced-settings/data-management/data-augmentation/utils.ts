// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ConfigurableParameterGroup, TrainingConfiguration } from '../../../../../../constants/shared-types';
import {
    findGroupByKey,
    isParameter,
    isParameterGroup,
} from '../../../../model-listing/model-training-parameters/utils';
import { isBoolEnableParameter } from '../../utils';

export type DataAugmentationConfigurationParameters = ConfigurableParameterGroup;

export const getDataAugmentationParameters = (trainingConfiguration: TrainingConfiguration) => {
    const datasetPreparation = findGroupByKey(trainingConfiguration.parameters, 'dataset_preparation')?.parameters;
    const dataAugmentation = findGroupByKey(datasetPreparation, 'augmentation');

    if (dataAugmentation?.parameters === undefined) return undefined;

    return {
        ...dataAugmentation,
        parameters: dataAugmentation.parameters.filter((parameter) => parameter.key !== 'tiling'),
    };
};

const getDeimFrameworkParameter = (parameters: ConfigurableParameterGroup['parameters']) => {
    return parameters.find((parameter) => isParameter(parameter) && parameter.key === 'deim_framework');
};

export const isDataAugmentationEnabled = (dataAugmentationParameters: DataAugmentationConfigurationParameters) => {
    const deimParameter = getDeimFrameworkParameter(dataAugmentationParameters.parameters);

    if (isParameter(deimParameter) && deimParameter.value === true) {
        return true;
    }

    return dataAugmentationParameters.parameters
        .map((group) => {
            if (isParameterGroup(group)) {
                const enableParameter = group.parameters.find(
                    (parameter) => isParameter(parameter) && isBoolEnableParameter(parameter)
                );

                return enableParameter;
            }

            return undefined;
        })
        .filter(Boolean)
        .some((parameter) => isParameter(parameter) && parameter.value === true);
};
