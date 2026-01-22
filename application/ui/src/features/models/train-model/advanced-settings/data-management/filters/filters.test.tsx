// Copyright (C) 2022-2025 Intel Corporation
// LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import { useState } from 'react';

import { fireEvent, screen } from '@testing-library/react';

import { TrainingConfiguration } from '../../../../../../../../core/configurable-parameters/services/configuration.interface';
import {
    getMockedConfigurationParameter,
    getMockedTrainingConfiguration,
} from '../../../../../../../../test-utils/mocked-items-factory/mocked-configuration-parameters';
import { providersRender as render } from '../../../../../../../../test-utils/required-providers-render';
import { Filters } from './filters.component';

type FiltersParameters = TrainingConfiguration['datasetPreparation']['filtering'];

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
    const filtersParameters = {
        min_annotation_pixels: [
            getMockedConfigurationParameter({
                key: 'enable',
                type: 'bool',
                name: 'Enable minimum annotation pixels filtering',
                value: false,
                description: 'Whether to apply minimum annotation pixels filtering',
                defaultValue: false,
            }),
            getMockedConfigurationParameter({
                key: 'min_annotation_pixels',
                type: 'int',
                name: 'Minimum annotation pixels',
                value: 1,
                description: 'Minimum number of pixels in an annotation',
                defaultValue: 1,
                maxValue: 200000000,
                minValue: 0,
            }),
        ],
        max_annotation_pixels: [
            getMockedConfigurationParameter({
                key: 'enable',
                type: 'bool',
                name: 'Enable maximum annotation pixels filtering',
                value: false,
                description: 'Whether to apply maximum annotation pixels filtering',
                defaultValue: false,
            }),
            getMockedConfigurationParameter({
                key: 'max_annotation_pixels',
                type: 'int',
                name: 'Maximum annotation pixels',
                value: 10000,
                description: 'Maximum number of pixels in an annotation',
                defaultValue: 10000,
                maxValue: null,
                minValue: 0,
            }),
        ],
        min_annotation_objects: [
            getMockedConfigurationParameter({
                key: 'enable',
                type: 'bool',
                name: 'Enable minimum annotation objects filtering',
                value: false,
                description: 'Whether to apply minimum annotation objects filtering',
                defaultValue: false,
            }),
            getMockedConfigurationParameter({
                key: 'min_annotation_objects',
                type: 'int',
                name: 'Minimum annotation objects',
                value: 1,
                description: 'Minimum number of objects in an annotation',
                defaultValue: 1,
                maxValue: null,
                minValue: 0,
            }),
        ],
        max_annotation_objects: [
            getMockedConfigurationParameter({
                key: 'enable',
                type: 'bool',
                name: 'Enable maximum annotation objects filtering',
                value: false,
                description: 'Whether to apply maximum annotation objects filtering',
                defaultValue: false,
            }),
            getMockedConfigurationParameter({
                key: 'max_annotation_objects',
                type: 'int',
                name: 'Maximum annotation objects',
                value: 10000,
                description: 'Maximum number of objects in an annotation',
                defaultValue: 10000,
                maxValue: null,
                minValue: 0,
            }),
        ],
    } satisfies FiltersParameters;

    const App = (props: { filtersParameters: FiltersParameters } = { filtersParameters }) => {
        const [trainingConfiguration, setTrainingConfiguration] = useState<TrainingConfiguration | undefined>(() =>
            getMockedTrainingConfiguration({
                datasetPreparation: {
                    filtering: props.filtersParameters,
                    augmentation: {},
                    subsetSplit: [],
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
                filtersConfiguration={trainingConfiguration?.datasetPreparation.filtering ?? props.filtersParameters}
                onUpdateTrainingConfiguration={handleUpdateTrainingConfiguration}
            />
        );
    };

    it('disables the filter option when unlimited checkbox is ticked', () => {
        render(<App filtersParameters={filtersParameters} />);

        Object.values(filtersParameters).forEach(([_enableParameter, configParameter]) => {
            expect(screen.getByText(configParameter.name)).toBeInTheDocument();
            expect(getToggleFilter(configParameter.name)).toBeChecked();
            expect(getFilterParameter(configParameter.name)).toBeDisabled();

            toggleFilter(configParameter.name);

            expect(getToggleFilter(configParameter.name)).not.toBeChecked();
            expect(getFilterParameter(configParameter.name)).toBeEnabled();
        });
    });

    it('changes the filter tag to "On" when at least one filter is enabled', () => {
        render(<App filtersParameters={filtersParameters} />);

        expect(screen.getByLabelText('Filters tag')).toHaveTextContent('Off');

        const filterParameter = filtersParameters.max_annotation_objects;
        toggleFilter(filterParameter[1].name);

        expect(screen.getByLabelText('Filters tag')).toHaveTextContent('On');
    });

    it('resets the parameter to default value', () => {
        render(<App filtersParameters={filtersParameters} />);

        Object.values(filtersParameters).forEach(([_enableParameter, configParameter]) => {
            toggleFilter(configParameter.name);

            expect(getToggleFilter(configParameter.name)).not.toBeChecked();

            fireEvent.click(screen.getByRole('button', { name: `Reset ${configParameter.name}` }));

            expect(getToggleFilter(configParameter.name)).toBeChecked();

            toggleFilter(configParameter.name);
            fireEvent.click(screen.getByRole('button', { name: `Increase Change ${configParameter.name}` }));

            const value = getFilterParameter(configParameter.name).getAttribute('value')?.replaceAll(',', '');
            expect(value).toBe(String(Number(configParameter.value) + 1));

            fireEvent.click(screen.getByRole('button', { name: `Reset ${configParameter.name}` }));

            const defaultValue = getFilterParameter(configParameter.name).getAttribute('value')?.replaceAll(',', '');

            expect(defaultValue).toBe(String(configParameter.defaultValue));
        });
    });
});
