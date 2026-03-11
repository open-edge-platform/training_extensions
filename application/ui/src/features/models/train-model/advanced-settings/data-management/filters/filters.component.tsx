// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import { Grid, minmax } from '@geti/ui';

import { ConfigurableParameterGroup, type TrainingConfiguration } from '../../../../../../constants/shared-types';
import { isParameterGroup } from '../../../../model-listing/model-training-parameters/utils';
import { Accordion } from '../../components/accordion/accordion.component';
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

const changeFilter = (
    trainingConfiguration: TrainingConfiguration,
    { key, newParameters }: { key: string; newParameters: FilterConfigurableParameters }
) => {
    const outputParameters = trainingConfiguration.parameters.map((parameterGroup) => {
        if (parameterGroup.key === 'dataset_preparation' && isParameterGroup(parameterGroup)) {
            return {
                ...parameterGroup,
                parameters: parameterGroup.parameters.map((parameters) => {
                    if (parameters.key === 'filtering' && isParameterGroup(parameters)) {
                        return {
                            ...parameters,
                            parameters: parameters.parameters.map((parameter) => {
                                if (isParameterGroup(parameter) && parameter.key === key) {
                                    return {
                                        ...parameter,
                                        parameters: newParameters,
                                    };
                                }

                                return parameter;
                            }),
                        };
                    }

                    return parameters;
                }),
            };
        }

        return parameterGroup;
    });

    return {
        parameters: outputParameters,
    };
};

export const Filters = ({ filtersParameters, onTrainingConfigurationChange }: FiltersProps) => {
    const handleFilterChange = (key: string, newParameters: FilterConfigurableParameters) => {
        onTrainingConfigurationChange((config) => {
            if (config === undefined) return;

            return changeFilter(config, { key, newParameters });
        });
    };

    const parameters = filtersParameters.parameters.filter(isParameterGroup).filter(isFilterConfigurableParameterGroup);

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
