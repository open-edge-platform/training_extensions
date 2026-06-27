// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@testing-library/react';
import { getMockedTrainingConfiguration } from 'mocks/mock-training-configuration';
import { describe, expect, it, vi } from 'vitest';

// eslint-disable-next-line no-restricted-imports
import { Provider, defaultTheme } from '@adobe/react-spectrum';
import { Evaluation } from './evaluation.component';

describe('Evaluation', () => {
    const mockOnTrainingConfigurationChange = vi.fn();
    const mockTrainingConfiguration = getMockedTrainingConfiguration();
    const mockDefaultTrainingConfiguration = getMockedTrainingConfiguration();

    it('renders evaluation parameters when present in configuration', async () => {
        render(
            <Provider theme={defaultTheme}>
                <Evaluation
                    trainingConfiguration={{ parameters: mockTrainingConfiguration }}
                    defaultTrainingConfiguration={{ parameters: mockDefaultTrainingConfiguration }}
                    onTrainingConfigurationChange={mockOnTrainingConfigurationChange}
                />
            </Provider>
        );

        expect(await screen.findByText('Evaluation parameters')).toBeInTheDocument();
    });

    it('does not render when evaluation parameters are missing', () => {
        const configWithoutEvaluation = {
            ...mockTrainingConfiguration,
            parameters: mockTrainingConfiguration.filter((p) => p.key !== 'evaluation'),
        };
        const defaultConfigWithoutEvaluation = {
            ...mockDefaultTrainingConfiguration,
            parameters: mockDefaultTrainingConfiguration.filter((p) => p.key !== 'evaluation'),
        };

        render(
            <Provider theme={defaultTheme}>
                <Evaluation
                    trainingConfiguration={{ parameters: configWithoutEvaluation.parameters }}
                    defaultTrainingConfiguration={{ parameters: defaultConfigWithoutEvaluation.parameters }}
                    onTrainingConfigurationChange={mockOnTrainingConfigurationChange}
                />
            </Provider>
        );

        expect(screen.queryByText('Evaluation parameters')).not.toBeInTheDocument();
    });
});
