// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import { View } from '@geti/ui';

import { TrainingConfiguration } from '../../../../../constants/shared-types';
import { TrainingSubsets } from './training-subsets/training-subsets.component';
import { getSubsetSplitParameters } from './training-subsets/utils';

type DataManagementProps = {
    trainingConfiguration: TrainingConfiguration;
    defaultTrainingConfiguration: TrainingConfiguration;
    onTrainingConfigurationChange: Dispatch<SetStateAction<TrainingConfiguration | undefined>>;
};

export const DataManagement = ({
    trainingConfiguration,
    defaultTrainingConfiguration,
    onTrainingConfigurationChange,
}: DataManagementProps) => {
    const subsetSplitParameters = getSubsetSplitParameters(trainingConfiguration);
    const defaultSubsetSplitParameters = getSubsetSplitParameters(defaultTrainingConfiguration);

    return (
        <View>
            {subsetSplitParameters !== undefined && (
                <TrainingSubsets
                    defaultSubsetParameters={defaultSubsetSplitParameters}
                    subsetsParameters={subsetSplitParameters}
                    onTrainingConfigurationChange={onTrainingConfigurationChange}
                />
            )}
        </View>
    );
};
