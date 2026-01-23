// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedConfigurationParameter, getMockedTrainingConfiguration } from 'mocks/mock-training-configuration';
import { render } from 'test-utils/render';

import { TrainingConfiguration } from '../../../../configuration.interface';
import { Filters } from './filters.component';

type FiltersParameters = TrainingConfiguration['dataset_preparation']['filtering'];

const getToggleFilter = (filterName: string) => {
    return screen.getByRole('checkbox', { name: `Toggle ${filterName}` });
};

const getFilterParameter = (filterName: string) => {
    return screen.getByRole('textbox', { name: `Change ${filterName}` });
};

const toggleFilter = async (filterName: string) => {
    await userEvent.click(getToggleFilter(filterName));
};

describe('Filters', () => {
    const filtersParameters = {
        min_annotation_pixels: [
            getMockedConfigurationParameter({
                key: 'enable',
                type: 'bool',
                name: 'Enable minimum annotation pixels filtering',
                value: false,
                description: 'Whether to apply minimum annotation pixels filtering',
                default_value: false,
            }),
            getMockedConfigurationParameter({
                key: 'min_annotation_pixels',
                type: 'int',
                name: 'Minimum annotation pixels',
                value: 1,
                description: 'Minimum number of pixels in an annotation',
                default_value: 1,
                max_value: 200000000,
                min_value: 0,
            }),
        ],
        max_annotation_pixels: [
            getMockedConfigurationParameter({
                key: 'enable',
                type: 'bool',
                name: 'Enable maximum annotation pixels filtering',
                value: false,
                description: 'Whether to apply maximum annotation pixels filtering',
                default_value: false,
            }),
            getMockedConfigurationParameter({
                key: 'max_annotation_pixels',
                type: 'int',
                name: 'Maximum annotation pixels',
                value: 10000,
                description: 'Maximum number of pixels in an annotation',
                default_value: 10000,
                max_value: null,
                min_value: 0,
            }),
        ],
        min_annotation_objects: [
            getMockedConfigurationParameter({
                key: 'enable',
                type: 'bool',
                name: 'Enable minimum annotation objects filtering',
                value: false,
                description: 'Whether to apply minimum annotation objects filtering',
                default_value: false,
            }),
            getMockedConfigurationParameter({
                key: 'min_annotation_objects',
                type: 'int',
                name: 'Minimum annotation objects',
                value: 1,
                description: 'Minimum number of objects in an annotation',
                default_value: 1,
                max_value: null,
                min_value: 0,
            }),
        ],
        max_annotation_objects: [
            getMockedConfigurationParameter({
                key: 'enable',
                type: 'bool',
                name: 'Enable maximum annotation objects filtering',
                value: false,
                description: 'Whether to apply maximum annotation objects filtering',
                default_value: false,
            }),
            getMockedConfigurationParameter({
                key: 'max_annotation_objects',
                type: 'int',
                name: 'Maximum annotation objects',
                value: 10000,
                description: 'Maximum number of objects in an annotation',
                default_value: 10000,
                max_value: null,
                min_value: 0,
            }),
        ],
    } satisfies FiltersParameters;

    const App = (props: { filtersParameters: FiltersParameters } = { filtersParameters }) => {
        const [trainingConfiguration, setTrainingConfiguration] = useState<TrainingConfiguration | undefined>(() =>
            getMockedTrainingConfiguration({
                dataset_preparation: {
                    filtering: props.filtersParameters,
                    augmentation: {},
                    subset_split: [],
                },
            })
        );

        const handleUpdateTrainingConfiguration = (
            updateFunction: (config: TrainingConfiguration | undefined) => TrainingConfiguration | undefined
        ) => {
            setTrainingConfiguration(updateFunction);
        };

        return (
            <Filters
                filtersConfiguration={trainingConfiguration?.dataset_preparation.filtering ?? props.filtersParameters}
                onUpdateTrainingConfiguration={handleUpdateTrainingConfiguration}
            />
        );
    };

    it('disables the filter option when unlimited checkbox is ticked', async () => {
        render(<App filtersParameters={filtersParameters} />);

        for (const [_enableParameter, configParameter] of Object.values(filtersParameters)) {
            expect(screen.getByText(configParameter.name)).toBeInTheDocument();
            expect(getToggleFilter(configParameter.name)).toBeChecked();
            expect(getFilterParameter(configParameter.name)).toBeDisabled();

            await toggleFilter(configParameter.name);

            expect(getToggleFilter(configParameter.name)).not.toBeChecked();
            expect(getFilterParameter(configParameter.name)).toBeEnabled();
        }
    });

    it('changes the filter tag to "On" when at least one filter is enabled', async () => {
        render(<App filtersParameters={filtersParameters} />);

        expect(screen.getByLabelText('Filters tag')).toHaveTextContent('Off');

        const filterParameter = filtersParameters.max_annotation_objects;
        await toggleFilter(filterParameter[1].name);

        expect(screen.getByLabelText('Filters tag')).toHaveTextContent('On');
    });

    it('resets the parameter to default value', async () => {
        render(<App filtersParameters={filtersParameters} />);

        for (const [_enableParameter, configParameter] of Object.values(filtersParameters)) {
            await toggleFilter(configParameter.name);

            expect(getToggleFilter(configParameter.name)).not.toBeChecked();

            await userEvent.click(screen.getByRole('button', { name: `Reset ${configParameter.name}` }));

            expect(getToggleFilter(configParameter.name)).toBeChecked();

            await toggleFilter(configParameter.name);
            await userEvent.click(screen.getByRole('button', { name: `Increase Change ${configParameter.name}` }));

            const value = getFilterParameter(configParameter.name).getAttribute('value')?.replaceAll(',', '');
            expect(value).toBe(String(Number(configParameter.value) + 1));

            await userEvent.click(screen.getByRole('button', { name: `Reset ${configParameter.name}` }));

            const defaultValue = getFilterParameter(configParameter.name).getAttribute('value')?.replaceAll(',', '');

            expect(defaultValue).toBe(String(configParameter.default_value));
        }
    });
});
