// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    ConfigurableParameter,
    ConfigurableParameterGroup,
    TrainingConfiguration,
} from '../../../../../../constants/shared-types';
import {
    findGroupByKey,
    isParameter,
    isParameterGroup,
} from '../../../../model-listing/model-training-parameters/utils';

type DataAugmentationConfigurationParameters = Omit<ConfigurableParameterGroup, 'parameters'> & {
    parameters: ConfigurableParameter[];
};

export type DataAugmentationConfigurableParameters = Omit<ConfigurableParameterGroup, 'parameters'> & {
    parameters: DataAugmentationConfigurationParameters[];
};

export const getDataAugmentationParameters = (
    trainingConfiguration: TrainingConfiguration
): DataAugmentationConfigurableParameters | undefined => {
    const datasetPreparation = findGroupByKey(trainingConfiguration.parameters, 'dataset_preparation')?.parameters;
    const dataAugmentation = findGroupByKey(datasetPreparation, 'augmentation');

    if (dataAugmentation?.parameters === undefined) return undefined;

    return {
        ...dataAugmentation,
        parameters: dataAugmentation.parameters
            .filter((parameter) => parameter.key !== 'tiling')
            .filter(isParameterGroup)
            .map((parameter) => ({
                ...parameter,
                parameters: parameter.parameters.filter(isParameter),
            })),
    };
};
