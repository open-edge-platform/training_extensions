// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import { Flex } from '@geti/ui';

import { type TrainingConfiguration } from '../../../../../../constants/shared-types';
import { Parameters } from '../../components/parameters.component';
import { DataAugmentationConfigurableParameters } from './utils';

type DataAugmentationParametersListProps = {
    dataAugmentationParameters: DataAugmentationConfigurableParameters;
    onTrainingConfigurationChange: Dispatch<SetStateAction<TrainingConfiguration | undefined>>;
};

export const DataAugmentationParametersList = ({
    dataAugmentationParameters,
    onTrainingConfigurationChange,
}: DataAugmentationParametersListProps) => {
    return (
        <Flex direction={'column'} gap={'size-300'}>
            {dataAugmentationParameters.parameters.map((parameters) => (
                <Parameters key={parameters.key} parameters={parameters} onChange={() => {}} />
            ))}
        </Flex>
    );
};
