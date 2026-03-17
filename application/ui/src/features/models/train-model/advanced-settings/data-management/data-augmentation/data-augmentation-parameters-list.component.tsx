// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import { ConfigurableParameter, type TrainingConfiguration } from '../../../../../../constants/shared-types';
import { ParametersGroup } from '../../components/parameters.component';
import { replaceByKey } from '../../utils';
import { DataAugmentationConfigurableParameters } from './utils';

type DataAugmentationParametersListProps = {
    dataAugmentationParameters: DataAugmentationConfigurableParameters;
    onTrainingConfigurationChange: Dispatch<SetStateAction<TrainingConfiguration | undefined>>;
};

const changeDataAugmentationParameters = (
    trainingConfiguration: TrainingConfiguration,
    parameterGroupKey: string,
    newParameter: ConfigurableParameter
): TrainingConfiguration => {
    const parameters: TrainingConfiguration['parameters'] = replaceByKey(
        trainingConfiguration.parameters,
        'dataset_preparation',
        (datasetPreparationGroup) => ({
            ...datasetPreparationGroup,
            parameters: replaceByKey(datasetPreparationGroup.parameters, 'augmentation', (augmentationGroup) => ({
                ...augmentationGroup,
                parameters: replaceByKey(augmentationGroup.parameters, parameterGroupKey, (parameterGroup) => ({
                    ...parameterGroup,
                    parameters: parameterGroup.parameters.map((parameter) =>
                        parameter.key === newParameter.key ? newParameter : parameter
                    ),
                })),
            })),
        })
    );

    return {
        parameters,
    };
};

export const DataAugmentationParametersList = ({
    dataAugmentationParameters,
    onTrainingConfigurationChange,
}: DataAugmentationParametersListProps) => {
    const handleAugmentationParameterChange = (groupKey: string, parameter: ConfigurableParameter) => {
        onTrainingConfigurationChange((config) => {
            if (config === undefined) return;

            return changeDataAugmentationParameters(config, groupKey, parameter);
        });
    };

    return (
        <ParametersGroup
            parameters={dataAugmentationParameters.parameters}
            onChange={handleAugmentationParameterChange}
        />
    );
};
