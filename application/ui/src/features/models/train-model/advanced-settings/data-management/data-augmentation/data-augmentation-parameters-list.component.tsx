// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import { ConfigurableParameter, type TrainingConfiguration } from '../../../../../../constants/shared-types';
import { ParametersGroup } from '../../components/parameters.component';
import { deepReplaceParameters } from '../../utils';
import { DataAugmentationConfigurableParameters } from './utils';

type DataAugmentationParametersListProps = {
    dataAugmentationParameters: DataAugmentationConfigurableParameters;
    onTrainingConfigurationChange: Dispatch<SetStateAction<TrainingConfiguration | undefined>>;
};

const changeDataAugmentationParameters = (
    trainingConfiguration: TrainingConfiguration,
    parameterGroupKeys: string[],
    newParameter: ConfigurableParameter
): TrainingConfiguration => {
    const parameters: TrainingConfiguration['parameters'] = deepReplaceParameters(
        trainingConfiguration.parameters,
        [newParameter],
        ['dataset_preparation', ...parameterGroupKeys]
    );

    return {
        parameters,
    };
};

export const DataAugmentationParametersList = ({
    dataAugmentationParameters,
    onTrainingConfigurationChange,
}: DataAugmentationParametersListProps) => {
    const handleAugmentationParameterChange = (parameter: ConfigurableParameter, groupKeys: string[] = []) => {
        onTrainingConfigurationChange((config) => {
            if (config === undefined) return;

            return changeDataAugmentationParameters(config, groupKeys, parameter);
        });
    };

    return (
        <ParametersGroup parametersGroup={dataAugmentationParameters} onChange={handleAugmentationParameterChange} />
    );
};
