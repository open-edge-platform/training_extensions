// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import { type TrainingConfiguration } from '../../../../../../constants/shared-types';
import { Accordion } from '../../components/accordion/accordion.component';
import { DataAugmentationParametersList } from './data-augmentation-parameters-list.component';
import { DataAugmentationConfigurationParameters, isDataAugmentationEnabled } from './utils';

type DataAugmentationProps = {
    dataAugmentationParameters: DataAugmentationConfigurationParameters;
    onTrainingConfigurationChange: Dispatch<SetStateAction<TrainingConfiguration | undefined>>;
};

export const DataAugmentation = ({
    dataAugmentationParameters,
    onTrainingConfigurationChange,
}: DataAugmentationProps) => {
    const isEnabled = isDataAugmentationEnabled(dataAugmentationParameters);

    return (
        <Accordion>
            <Accordion.Title>
                Data Augmentation
                <Accordion.Tag ariaLabel={'Data augmentation tag'}>{isEnabled ? 'Yes' : 'No'}</Accordion.Tag>
            </Accordion.Title>
            <Accordion.Content>
                <Accordion.Description>{dataAugmentationParameters.description}</Accordion.Description>
                <Accordion.Divider marginY={'size-250'} />
                <DataAugmentationParametersList
                    dataAugmentationParameters={dataAugmentationParameters}
                    onTrainingConfigurationChange={onTrainingConfigurationChange}
                />
            </Accordion.Content>
        </Accordion>
    );
};
