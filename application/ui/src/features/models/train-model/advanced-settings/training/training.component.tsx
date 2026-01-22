// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { View } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { TrainingConfiguration } from '../../../configuration.interface';
import { FineTuneParameters } from './fine-tune-parameters.component';
import { LearningParameters } from './learning-parameters/learning-parameters.component';

interface TrainingProps {
    trainFromScratch: boolean;
    onTrainFromScratchChange: (trainFromScratch: boolean) => void;

    trainingConfiguration: TrainingConfiguration;
    defaultTrainingConfiguration: TrainingConfiguration;
    onUpdateTrainingConfiguration: (
        updateFunction: (config: TrainingConfiguration | undefined) => TrainingConfiguration | undefined
    ) => void;

    isReshufflingSubsetsEnabled: boolean;
    onReshufflingSubsetsEnabledChange: (reshufflingSubsetsEnabled: boolean) => void;
}

export const Training = ({
    trainFromScratch,
    onTrainFromScratchChange,
    trainingConfiguration,
    onReshufflingSubsetsEnabledChange,
    isReshufflingSubsetsEnabled,
    onUpdateTrainingConfiguration,
    defaultTrainingConfiguration,
}: TrainingProps) => {
    return (
        <View>
            <FineTuneParameters
                trainFromScratch={trainFromScratch}
                onTrainFromScratchChange={onTrainFromScratchChange}
                isReshufflingSubsetsEnabled={isReshufflingSubsetsEnabled}
                onReshufflingSubsetsEnabledChange={onReshufflingSubsetsEnabledChange}
            />
            {!isEmpty(trainingConfiguration.training) && (
                <LearningParameters
                    defaultParameters={defaultTrainingConfiguration.training}
                    parameters={trainingConfiguration.training}
                    onUpdateTrainingConfiguration={onUpdateTrainingConfiguration}
                />
            )}
        </View>
    );
};
