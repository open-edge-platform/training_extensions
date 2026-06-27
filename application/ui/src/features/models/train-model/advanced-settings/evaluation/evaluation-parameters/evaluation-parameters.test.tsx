// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@testing-library/react';
import { getMockedConfigurationParameterGroup } from 'mocks/mock-training-configuration';
import { describe, expect, it, vi } from 'vitest';

// eslint-disable-next-line no-restricted-imports
import { Provider, defaultTheme } from '@adobe/react-spectrum';
import { EvaluationParameters } from './evaluation-parameters.component';
import { EvaluationConfigurationGroup } from './utils';

describe('EvaluationParameters', () => {
    const mockOnTrainingConfigurationChange = vi.fn();

    const evaluationParameters: EvaluationConfigurationGroup = getMockedConfigurationParameterGroup({
        key: 'evaluation',
        name: 'Evaluation parameters',
        description: 'Test evaluation description',
        parameters: [],
    });

    const defaultEvaluationParameters: EvaluationConfigurationGroup = getMockedConfigurationParameterGroup({
        key: 'evaluation',
        name: 'Evaluation parameters',
        description: 'Test evaluation description',
        parameters: [],
    });

    it('renders with "Default" tag when parameters match default', async () => {
        render(
            <Provider theme={defaultTheme}>
                <EvaluationParameters
                    evaluationParameters={evaluationParameters}
                    defaultEvaluationParameters={defaultEvaluationParameters}
                    onTrainingConfigurationChange={mockOnTrainingConfigurationChange}
                />
            </Provider>
        );

        expect(await screen.findByText('Evaluation parameters')).toBeInTheDocument();
        expect(await screen.findByLabelText('Evaluation parameters tag')).toHaveTextContent('Default');
    });

    it('renders with "Modified" tag when parameters do not match default', async () => {
        const modifiedParameters: EvaluationConfigurationGroup = {
            ...evaluationParameters,
            description: 'Modified description',
        };

        render(
            <Provider theme={defaultTheme}>
                <EvaluationParameters
                    evaluationParameters={modifiedParameters}
                    defaultEvaluationParameters={defaultEvaluationParameters}
                    onTrainingConfigurationChange={mockOnTrainingConfigurationChange}
                />
            </Provider>
        );

        expect(await screen.findByLabelText('Evaluation parameters tag')).toHaveTextContent('Modified');
    });
});
