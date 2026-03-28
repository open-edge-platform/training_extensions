// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import { Grid, minmax } from '@geti/ui';

import type { ConfigurableParameterGroup, TrainingConfiguration } from '../../../../../../constants/shared-types';
import { Accordion } from '../../components/accordion/accordion.component';
import { deepReplaceParameters } from '../../utils';
import { FiltersOptions } from './filters-options.component';
import {
    checkIfFiltersAreEnabled,
    isFilterConfigurableParameterGroup,
    type FilterConfigurableParameters,
} from './utils';

type FiltersProps = {
    filtersParameters: ConfigurableParameterGroup;
    onTrainingConfigurationChange: Dispatch<SetStateAction<TrainingConfiguration | undefined>>;
};

const changeFilterParameters = (
    trainingConfiguration: TrainingConfiguration,
    { key, newParameters }: { key: string; newParameters: FilterConfigurableParameters }
): TrainingConfiguration => ({
    parameters: deepReplaceParameters(trainingConfiguration.parameters, newParameters, [
        'dataset_preparation',
        'filtering',
        key,
    ]),
});

export const Filters = ({ filtersParameters, onTrainingConfigurationChange }: FiltersProps) => {
    const handleFilterChange = (key: string, newParameters: FilterConfigurableParameters) => {
        onTrainingConfigurationChange((config) => {
            if (config === undefined) return;

            return changeFilterParameters(config, { key, newParameters });
        });
    };

    const parameters = filtersParameters.parameters.filter(isFilterConfigurableParameterGroup);

    const areFiltersEnabled = checkIfFiltersAreEnabled(parameters);

    return (
        <Accordion>
            <Accordion.Title>
                Filters <Accordion.Tag ariaLabel={'Filters tag'}>{areFiltersEnabled ? 'On' : 'Off'}</Accordion.Tag>
            </Accordion.Title>
            <Accordion.Content>
                <Accordion.Description>{filtersParameters.description}</Accordion.Description>
                <Accordion.Divider marginY={'size-250'} />
                <Grid
                    columns={['size-3000', minmax('size-3400', '1fr'), 'size-400']}
                    gap={'size-300'}
                    alignItems={'center'}
                >
                    <FiltersOptions filterParameters={parameters} onFilterChange={handleFilterChange} />
                </Grid>
            </Accordion.Content>
        </Accordion>
    );
};
