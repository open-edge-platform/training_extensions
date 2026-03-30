// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import { View } from '@geti-ui/ui';

import { TrainingConfiguration } from '../../../../../constants/shared-types';
import { LearningParameters } from './learning-parameters/learning-parameters.component';
import { getLearningParameters } from './learning-parameters/utils';

type TrainingProps = {
    trainingConfiguration: TrainingConfiguration;
    defaultTrainingConfiguration: TrainingConfiguration;
    onTrainingConfigurationChange: Dispatch<SetStateAction<TrainingConfiguration | undefined>>;
};

export const Training = ({
    trainingConfiguration,
    defaultTrainingConfiguration,
    onTrainingConfigurationChange,
}: TrainingProps) => {
    const learningParameters = getLearningParameters(trainingConfiguration);
    const defaultLearningParameters = getLearningParameters(defaultTrainingConfiguration);

    return (
        <View>
            {learningParameters !== undefined && (
                <LearningParameters
                    learningParameters={learningParameters}
                    defaultLearningParameters={defaultLearningParameters}
                    onTrainingConfigurationChange={onTrainingConfigurationChange}
                />
            )}
        </View>
    );
};
