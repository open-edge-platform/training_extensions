// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { TrainingConfiguration } from '../../../../configuration.interface';
import { Accordion } from '../../ui/accordion/accordion.component';
import { isBoolEnableParameter } from '../../utils';
import {
    DataAugmentationParameters,
    DataAugmentationParametersList,
} from './data-augmentation-parameters-list.component';

type DataAugmentationProps = {
    parameters: DataAugmentationParameters;
    onUpdateTrainingConfiguration: (
        updateFunction: (config: TrainingConfiguration | undefined) => TrainingConfiguration | undefined
    ) => void;
};

export const isDataAugmentationEnabled = (parameters: DataAugmentationParameters): boolean => {
    return Object.values(parameters).some((parametersGroup) =>
        parametersGroup.some((parameter) => isBoolEnableParameter(parameter) && parameter.value === true)
    );
};

export const DataAugmentation = ({ parameters, onUpdateTrainingConfiguration }: DataAugmentationProps) => {
    const isEnabled = isDataAugmentationEnabled(parameters);

    return (
        <Accordion>
            <Accordion.Title>
                Data Augmentation
                <Accordion.Tag ariaLabel={'Data augmentation tag'}>{isEnabled ? 'Yes' : 'No'}</Accordion.Tag>
            </Accordion.Title>
            <Accordion.Content>
                <Accordion.Description>
                    Choose data augmentation transformations to enhance the diversity of available data for training
                    models.
                </Accordion.Description>
                <Accordion.Divider marginY={'size-250'} />
                <DataAugmentationParametersList
                    parameters={parameters}
                    onUpdateTrainingConfiguration={onUpdateTrainingConfiguration}
                />
            </Accordion.Content>
        </Accordion>
    );
};
