// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction, useMemo } from 'react';

import { ConfigurableParameter, type TrainingConfiguration } from '../../../../../../constants/shared-types';
import { Parameters } from '../../components/parameters.component';
import { deepReplaceParameters, filterDependentParameters } from '../../utils';
import { DataAugmentationConfigurationParameters } from './utils';

type DataAugmentationParametersListProps = {
    dataAugmentationParameters: DataAugmentationConfigurationParameters;
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
        ['dataset_preparation', 'augmentation', ...parameterGroupKeys]
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

    const augmentationParameters = useMemo(() => {
        return filterDependentParameters(dataAugmentationParameters.parameters);
    }, [dataAugmentationParameters.parameters]);

    return <Parameters parameters={augmentationParameters} onChange={handleAugmentationParameterChange} />;
};
