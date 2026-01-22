// Copyright (C) 2022-2025 Intel Corporation
// LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import { screen } from '@testing-library/react';

import { TrainingConfiguration } from '../../../../../../../core/configurable-parameters/services/configuration.interface';
import { providersRender as render } from '../../../../../../../test-utils/required-providers-render';
import { Training } from './training.component';

const baseConfig: TrainingConfiguration = {
    training: [
        {
            key: 'max_epochs',
            name: 'Maximum epochs',
            type: 'int',
            description: 'Maximum number of training epochs to run',
            value: 100,
            defaultValue: 100,
            minValue: 0,
            maxValue: null,
        },
    ],
    datasetPreparation: {
        subsetSplit: [],
        augmentation: {},
        filtering: {},
    },
    taskId: '',
    evaluation: [],
};

describe('Training component', () => {
    it('renders LearningParameters if training parameters are not empty', () => {
        render(
            <Training
                trainFromScratch={false}
                onTrainFromScratchChange={jest.fn()}
                defaultTrainingConfiguration={baseConfig}
                trainingConfiguration={baseConfig}
                onUpdateTrainingConfiguration={jest.fn()}
                isReshufflingSubsetsEnabled={true}
                onReshufflingSubsetsEnabledChange={jest.fn()}
            />
        );

        expect(screen.getByLabelText('Learning parameters tag')).toBeInTheDocument();
    });

    it('does not render LearningParameters if training parameters are empty', () => {
        render(
            <Training
                trainFromScratch={false}
                onTrainFromScratchChange={jest.fn()}
                trainingConfiguration={{ ...baseConfig, training: [] }}
                defaultTrainingConfiguration={{ ...baseConfig, training: [] }}
                onUpdateTrainingConfiguration={jest.fn()}
                isReshufflingSubsetsEnabled={true}
                onReshufflingSubsetsEnabledChange={jest.fn()}
            />
        );

        expect(screen.queryByLabelText('Learning parameters tag')).not.toBeInTheDocument();
    });
});
