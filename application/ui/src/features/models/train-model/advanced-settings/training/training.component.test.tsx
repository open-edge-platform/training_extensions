// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { render } from 'test-utils/render';
import { vi } from 'vitest';

import { TrainingConfiguration } from '../../../configuration.interface';
import { Training } from './training.component';

const baseConfig: TrainingConfiguration = {
    training: [
        {
            key: 'max_epochs',
            name: 'Maximum epochs',
            type: 'int',
            description: 'Maximum number of training epochs to run',
            value: 100,
            default_value: 100,
            min_value: 0,
            max_value: null,
        },
    ],
    dataset_preparation: {
        subset_split: [],
        augmentation: {},
        filtering: {},
    },
    evaluation: [],
};

describe('Training component', () => {
    it('renders LearningParameters if training parameters are not empty', () => {
        render(
            <Training
                trainFromScratch={false}
                onTrainFromScratchChange={vi.fn()}
                defaultTrainingConfiguration={baseConfig}
                trainingConfiguration={baseConfig}
                onUpdateTrainingConfiguration={vi.fn()}
                isReshufflingSubsetsEnabled={true}
                onReshufflingSubsetsEnabledChange={vi.fn()}
            />
        );

        expect(screen.getByLabelText('Learning parameters tag')).toBeInTheDocument();
    });

    it('does not render LearningParameters if training parameters are empty', () => {
        render(
            <Training
                trainFromScratch={false}
                onTrainFromScratchChange={vi.fn()}
                trainingConfiguration={{ ...baseConfig, training: [] }}
                defaultTrainingConfiguration={{ ...baseConfig, training: [] }}
                onUpdateTrainingConfiguration={vi.fn()}
                isReshufflingSubsetsEnabled={true}
                onReshufflingSubsetsEnabledChange={vi.fn()}
            />
        );

        expect(screen.queryByLabelText('Learning parameters tag')).not.toBeInTheDocument();
    });
});
