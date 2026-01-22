// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';

import { ConfigurationParameter, TrainingConfiguration } from '../../../../configuration.interface';
import { Parameters } from '../../ui/parameters.component';

export type DataAugmentationParameters = TrainingConfiguration['dataset_preparation']['augmentation'];

type DataAugmentationParametersListProps = {
    parameters: DataAugmentationParameters;
    onUpdateTrainingConfiguration: (
        updateFunction: (config: TrainingConfiguration | undefined) => TrainingConfiguration | undefined
    ) => void;
    isReadOnly?: boolean;
};

export const DataAugmentationParametersList = ({
    parameters,
    onUpdateTrainingConfiguration,
    isReadOnly = false,
}: DataAugmentationParametersListProps) => {
    const handleChange = (key: string) => (inputParameter: ConfigurationParameter) => {
        onUpdateTrainingConfiguration((config) => {
            if (!config) return undefined;

            const newConfig = structuredClone(config);
            newConfig.dataset_preparation.augmentation[key] = config.dataset_preparation.augmentation[key].map(
                (parameter) => (parameter.key === inputParameter.key ? inputParameter : parameter)
            );

            return newConfig;
        });
    };

    return (
        <Flex direction={'column'} height={'size-100%'} gap={'size-300'}>
            {Object.entries(parameters).map(([key, parametersLocal]) => {
                return (
                    <Parameters
                        key={key}
                        parameters={parametersLocal}
                        onChange={handleChange(key)}
                        isReadOnly={isReadOnly}
                    />
                );
            })}
        </Flex>
    );
};
