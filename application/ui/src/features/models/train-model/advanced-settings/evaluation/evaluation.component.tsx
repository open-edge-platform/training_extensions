// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import { View } from '@geti/ui';

import { TrainingConfiguration } from '../../../../../constants/shared-types';
import { EvaluationParameters } from './evaluation-parameters/evaluation-parameters.component';
import { getEvaluationParameters } from './evaluation-parameters/utils';

type EvaluationProps = {
    trainingConfiguration: TrainingConfiguration;
    defaultTrainingConfiguration: TrainingConfiguration;
    onTrainingConfigurationChange: Dispatch<SetStateAction<TrainingConfiguration | undefined>>;
};

export const Evaluation = ({
    trainingConfiguration,
    defaultTrainingConfiguration,
    onTrainingConfigurationChange,
}: EvaluationProps) => {
    const evaluationParameters = getEvaluationParameters(trainingConfiguration);
    const defaultEvaluationParameters = getEvaluationParameters(defaultTrainingConfiguration);

    return (
        <View>
            {evaluationParameters !== undefined && (
                <EvaluationParameters
                    evaluationParameters={evaluationParameters}
                    defaultEvaluationParameters={defaultEvaluationParameters}
                    onTrainingConfigurationChange={onTrainingConfigurationChange}
                />
            )}
        </View>
    );
};
