// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {
    getMockedConfigurationParameter,
    getMockedConfigurationParameterGroup,
} from 'mocks/mock-training-configuration';
import { render } from 'test-utils/render';

import { ConfigurableParameterGroup, TrainingConfiguration } from '../../../../../../constants/shared-types';
import { IntensityMapping } from './intensity-mapping.component';
import { getIntensityMappingParameters } from './utils';

describe('IntensityMapping', () => {
    const intensityMappingParameters: ConfigurableParameterGroup = getMockedConfigurationParameterGroup({
        key: 'intensity_mapping',
        name: 'Intensity mapping',
        description:
            'Intensity mapping parameters control how raw pixel values are normalised to [0, 1] range before training.',
        parameters: [
            getMockedConfigurationParameter({
                key: 'mode',
                value_type: 'str',
                name: 'Intensity mapping mode',
                value: 'Unit interval scaling',
                default_value: 'Unit interval scaling',
                allowed_values: ['Unit interval scaling', 'Windowing', 'Range scaling with clipping'],
                depends_on: null,
            }),
            getMockedConfigurationParameter({
                key: 'max_intensity_value',
                value_type: 'float',
                name: 'Maximum pixel intensity',
                value: 255.0,
                default_value: 255.0,
                min_value: 0,
                max_value: null,
                allowed_values: null,
                depends_on: { mode: 'Unit interval scaling' },
            }),
            getMockedConfigurationParameter({
                key: 'window_center',
                value_type: 'float',
                name: 'Window center',
                value: 127.5,
                default_value: 127.5,
                min_value: null,
                max_value: null,
                allowed_values: null,
                depends_on: { mode: 'Windowing' },
            }),
            getMockedConfigurationParameter({
                key: 'window_width',
                value_type: 'float',
                name: 'Window width',
                value: 255.0,
                default_value: 255.0,
                min_value: 0,
                max_value: null,
                allowed_values: null,
                depends_on: { mode: 'Windowing' },
            }),
        ],
    });

    const App = () => {
        const [trainingConfiguration, setTrainingConfiguration] = useState<TrainingConfiguration | undefined>({
            parameters: [
                getMockedConfigurationParameterGroup({
                    key: 'dataset_preparation',
                    parameters: [intensityMappingParameters],
                }),
            ],
        });

        const currentParameters =
            getIntensityMappingParameters(trainingConfiguration ?? { parameters: [] }) ?? intensityMappingParameters;

        return (
            <IntensityMapping
                intensityMappingParameters={currentParameters}
                onTrainingConfigurationChange={setTrainingConfiguration}
            />
        );
    };

    it('renders the intensity mapping section with description', () => {
        render(<App />);

        expect(screen.getByText('Intensity Mapping')).toBeInTheDocument();
        expect(
            screen.getByText(
                'Intensity mapping parameters control how raw pixel values are normalised to [0, 1] range before training.'
            )
        ).toBeInTheDocument();
    });

    it('shows mode-dependent parameters based on selected mode', () => {
        render(<App />);

        // In "Unit interval scaling" mode, max_intensity_value should be visible
        expect(screen.getByRole('textbox', { name: 'Change Maximum pixel intensity' })).toBeInTheDocument();

        // Windowing parameters should not be visible
        expect(screen.queryByRole('textbox', { name: 'Change Window center' })).not.toBeInTheDocument();
        expect(screen.queryByRole('textbox', { name: 'Change Window width' })).not.toBeInTheDocument();
    });

    it('updates parameter value when changed', async () => {
        render(<App />);

        const maxIntensityInput = screen.getByRole('textbox', { name: 'Change Maximum pixel intensity' });
        expect(maxIntensityInput).toHaveValue('255');

        await userEvent.click(screen.getByRole('button', { name: 'Increase Change Maximum pixel intensity' }));

        expect(maxIntensityInput).toHaveValue('255.1');
    });
});
