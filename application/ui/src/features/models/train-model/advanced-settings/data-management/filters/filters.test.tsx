// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { fireEvent, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {
    getMockedConfigurationParameter,
    getMockedConfigurationParameterGroup,
} from 'mocks/mock-training-configuration';
import { render } from 'test-utils/render';

import { ConfigurableParameterGroup, TrainingConfiguration } from '../../../../../../constants/shared-types';
import { Filters } from './filters.component';
import { getFiltersParameters, isFilterConfigurableParameterGroup } from './utils';

const getToggleFilter = (filterName: string) => {
    return screen.getByRole('checkbox', { name: `Toggle ${filterName}` });
};

const getFilterParameter = (filterName: string) => {
    return screen.getByRole('textbox', { name: `Change ${filterName}` });
};

const toggleFilter = (filterName: string) => {
    fireEvent.click(getToggleFilter(filterName));
};

describe('Filters', () => {
    const filtersParameters = getMockedConfigurationParameterGroup({
        key: 'filtering',
        description:
            'Filtering parameters define criteria for including or excluding annotations from the dataset. Depending on the scenario, an appropriate filter configuration can speed up the training process and/or improve the model performance by removing noisy annotations.',
        parameters: [
            getMockedConfigurationParameterGroup({
                key: 'min_annotation_pixels',
                name: 'Minimum annotation pixels',
                parameters: [
                    getMockedConfigurationParameter({
                        key: 'enable',
                        value_type: 'bool',
                        name: 'Enable minimum annotation pixels filtering',
                        value: false,
                        description: 'Whether to apply minimum annotation pixels filtering',
                        default_value: false,
                    }),
                    getMockedConfigurationParameter({
                        key: 'value',
                        value_type: 'int',
                        name: 'Minimum annotation pixels',
                        value: 1,
                        description: 'Minimum number of pixels in an annotation',
                        default_value: 1,
                        max_value: 200000000,
                        min_value: 0,
                    }),
                ],
            }),
            getMockedConfigurationParameterGroup({
                key: 'min_annotation_objects',
                name: 'Minimum annotation objects',
                parameters: [
                    getMockedConfigurationParameter({
                        key: 'enable',
                        value_type: 'bool',
                        name: 'Enable minimum annotation objects filtering',
                        value: false,
                        description: 'Whether to apply minimum annotation objects filtering',
                        default_value: false,
                    }),
                    getMockedConfigurationParameter({
                        key: 'value',
                        value_type: 'int',
                        name: 'Minimum annotation objects',
                        value: 10000,
                        description: 'Minimum number of objects in an annotation',
                        default_value: 10000,
                        max_value: null,
                        min_value: 0,
                    }),
                ],
            }),
            getMockedConfigurationParameterGroup({
                key: 'max_annotation_objects',
                name: 'Maximum annotation objects',
                parameters: [
                    getMockedConfigurationParameter({
                        key: 'enable',
                        value_type: 'bool',
                        name: 'Enable maximum annotation objects filtering',
                        value: false,
                        description: 'Whether to apply maximum annotation objects filtering',
                        default_value: false,
                    }),
                    getMockedConfigurationParameter({
                        key: 'value',
                        value_type: 'int',
                        name: 'Maximum annotation objects',
                        value: 1,
                        description: 'Maximum number of objects in an annotation',
                        default_value: 1,
                        max_value: null,
                        min_value: 0,
                    }),
                ],
            }),
        ],
    });

    const App = (props: { filtersParameters: ConfigurableParameterGroup } = { filtersParameters }) => {
        const [trainingConfiguration, setTrainingConfiguration] = useState<TrainingConfiguration | undefined>({
            parameters: [
                getMockedConfigurationParameterGroup({
                    key: 'dataset_preparation',
                    parameters: [filtersParameters],
                }),
            ],
        });

        return (
            <Filters
                filtersParameters={
                    getFiltersParameters(trainingConfiguration ?? { parameters: [] }) ?? props.filtersParameters
                }
                onTrainingConfigurationChange={setTrainingConfiguration}
            />
        );
    };

    it('disables the filter option when "no minimum/no maximum" checkbox is ticked', () => {
        render(<App filtersParameters={filtersParameters} />);

        const parameters = filtersParameters.parameters.filter(isFilterConfigurableParameterGroup);

        parameters.forEach((parameterGroup) => {
            const [_enableParameter, configurableParameter] = parameterGroup.parameters;

            expect(screen.getByText(configurableParameter.name)).toBeInTheDocument();
            expect(getToggleFilter(configurableParameter.name)).toBeChecked();
            expect(getFilterParameter(configurableParameter.name)).toBeDisabled();

            toggleFilter(configurableParameter.name);

            expect(getToggleFilter(configurableParameter.name)).not.toBeChecked();
            expect(getFilterParameter(configurableParameter.name)).toBeEnabled();
        });
    });

    it('displays "no minimum" for min and "no maximum" for max parameter', () => {
        render(<App filtersParameters={filtersParameters} />);

        const parameters = filtersParameters.parameters.filter(isFilterConfigurableParameterGroup);

        parameters.forEach((parameterGroup) => {
            const [_enableParameter, configurableParameter] = parameterGroup.parameters;

            const toggle = getToggleFilter(configurableParameter.name).parentElement as HTMLElement;

            if (parameterGroup.key.includes('min')) {
                expect(within(toggle).getByText('No minimum')).toBeInTheDocument();
            } else if (parameterGroup.key.includes('max')) {
                expect(within(toggle).getByText('No maximum')).toBeInTheDocument();
            }
        });
    });

    it('changes the filter tag to "On" when at least one filter is enabled', () => {
        render(<App filtersParameters={filtersParameters} />);

        const [minAnnotationsPixels] = filtersParameters.parameters.filter(isFilterConfigurableParameterGroup);
        const [_enableMinAnnotationPixels, configureMinAnnotationsPixels] = minAnnotationsPixels.parameters;

        expect(screen.getByLabelText('Filters tag')).toHaveTextContent('Off');

        expect(getToggleFilter(configureMinAnnotationsPixels.name)).toBeChecked();

        toggleFilter(configureMinAnnotationsPixels.name);

        expect(getToggleFilter(configureMinAnnotationsPixels.name)).not.toBeChecked();

        expect(screen.getByLabelText('Filters tag')).toHaveTextContent('On');
    });

    it('resets the parameter to default value', async () => {
        render(<App filtersParameters={filtersParameters} />);

        const parameters = filtersParameters.parameters.filter(isFilterConfigurableParameterGroup);

        for (const parameterGroup of parameters) {
            const [_enableParameter, configurableParameter] = parameterGroup.parameters;

            toggleFilter(configurableParameter.name);

            expect(getToggleFilter(configurableParameter.name)).not.toBeChecked();

            await userEvent.click(screen.getByRole('button', { name: `Reset ${configurableParameter.name}` }));

            expect(getToggleFilter(configurableParameter.name)).toBeChecked();

            toggleFilter(configurableParameter.name);

            const parameterInput = getFilterParameter(configurableParameter.name);
            await userEvent.clear(parameterInput);
            await userEvent.type(parameterInput, String(Number(configurableParameter.value) + 1));

            const value = getFilterParameter(configurableParameter.name).getAttribute('value')?.replaceAll(',', '');
            expect(value).toBe(String(Number(configurableParameter.value) + 1));

            await userEvent.click(screen.getByRole('button', { name: `Reset ${configurableParameter.name}` }));

            const defaultValue = getFilterParameter(configurableParameter.name)
                .getAttribute('value')
                ?.replaceAll(',', '');

            expect(defaultValue).toBe(String(configurableParameter.default_value));
        }
    });
});
