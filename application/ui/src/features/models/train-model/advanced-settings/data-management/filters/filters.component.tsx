// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Grid, minmax } from '@geti/ui';

import { BoolParameter, NumberParameter, TrainingConfiguration } from '../../../../configuration.interface';
import { Accordion } from '../../ui/accordion/accordion.component';
import { isBoolParameter, isNumberParameter } from '../../utils';
import { FilterOption, FiltersOptions } from './filters-options.component';

type FiltersConfiguration = TrainingConfiguration['dataset_preparation']['filtering'];

type FiltersProps = {
    filtersConfiguration: FiltersConfiguration;
    onUpdateTrainingConfiguration: (
        updateFunction: (config: TrainingConfiguration | undefined) => TrainingConfiguration | undefined
    ) => void;
};

export const Filters = ({ filtersConfiguration, onUpdateTrainingConfiguration }: FiltersProps) => {
    const filterOptions: FilterOption[] = Object.values(filtersConfiguration)
        .map(([enableParameter, configParameter]) => {
            if (isBoolParameter(enableParameter) && isNumberParameter(configParameter)) {
                return [enableParameter, configParameter] as [BoolParameter, NumberParameter];
            }

            return undefined;
        })
        .filter((option) => option !== undefined);

    const handleFilterOptionChange = ([enableParameter, configParameter]: FilterOption) => {
        onUpdateTrainingConfiguration((prevConfig) => {
            if (prevConfig === undefined) {
                return;
            }

            const newConfig = structuredClone(prevConfig);

            newConfig.dataset_preparation.filtering = Object.entries(
                prevConfig.dataset_preparation.filtering
            ).reduce<FiltersConfiguration>((acc, [key, parameters]) => {
                const [enableParameterLocal, configParameterLocal] = parameters;

                if (
                    enableParameter.key === enableParameterLocal.key &&
                    configParameter.key === configParameterLocal.key
                ) {
                    return {
                        ...acc,
                        [key]: [enableParameter, configParameter],
                    };
                }

                return {
                    ...acc,
                    [key]: parameters,
                };
            }, {});

            return newConfig;
        });
    };

    const areFiltersEnabled = filterOptions.some(([enableParameter]) => enableParameter.value);

    return (
        <Accordion>
            <Accordion.Title>
                Filters <Accordion.Tag ariaLabel={'Filters tag'}>{areFiltersEnabled ? 'On' : 'Off'}</Accordion.Tag>
            </Accordion.Title>
            <Accordion.Content>
                <Accordion.Description>
                    Use filters to specify the criteria for annotations that will be used for training.
                </Accordion.Description>
                <Accordion.Divider marginY={'size-250'} />
                <Grid
                    columns={['size-3000', minmax('size-3400', '1fr'), 'size-400']}
                    gap={'size-300'}
                    alignItems={'center'}
                >
                    <FiltersOptions options={filterOptions} onOptionsChange={handleFilterOptionChange} />
                </Grid>
            </Accordion.Content>
        </Accordion>
    );
};
