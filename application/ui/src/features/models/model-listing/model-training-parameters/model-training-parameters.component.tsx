// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Fragment } from 'react';

import { Grid, Text } from '@geti/ui';

import { TrainingConfigurationParameter } from '../../../../constants/shared-types';
import { useGetModelTrainingConfiguration } from '../../hooks/api/use-get-model-training-configuration.hook';
import { filterDependentParameters } from '../../train-model/advanced-settings/utils';
import { Box } from '../components/box/box.component';
import { findGroupByKey, flattenParameters } from './utils';

import classes from './model-training.module.scss';

type ModelTrainingParametersProps = {
    modelId: string;
};

type TrainingConfigurationParametersListProps = {
    parameters: TrainingConfigurationParameter[];
};

const TrainingConfigurationParametersList = ({ parameters }: TrainingConfigurationParametersListProps) => {
    const parameterRows = flattenParameters(parameters);

    if (parameterRows.length === 0) {
        return <Text>No parameters.</Text>;
    }

    return (
        <Grid columns={['1fr', '1fr']} gap={'size-100'}>
            {parameterRows.map((row, index) => (
                <Fragment key={`${index}-${row.isGroup}-${row.depth}-${row.name}-${row.value}`}>
                    <Text
                        UNSAFE_style={{
                            paddingInlineStart: `calc(${row.depth} * var(--spectrum-global-dimension-size-200))`,
                        }}
                    >
                        {row.isGroup ? row.name : `• ${row.name}`}
                    </Text>

                    <Text>{row.isGroup ? '' : row.value}</Text>
                </Fragment>
            ))}
        </Grid>
    );
};

export const ModelTrainingParameters = ({ modelId }: ModelTrainingParametersProps) => {
    const { data } = useGetModelTrainingConfiguration(modelId);

    const trainingGroup = findGroupByKey(data?.parameters, 'training');
    const datasetPreparationGroup = findGroupByKey(data?.parameters, 'dataset_preparation');
    const filteringGroup = findGroupByKey(datasetPreparationGroup?.parameters, 'filtering');
    const augmentationGroup = findGroupByKey(datasetPreparationGroup?.parameters, 'augmentation');

    const learningParameters = filterDependentParameters(trainingGroup?.parameters ?? []);

    return (
        <Grid columns={['1fr', '1fr', '1fr']} gap={'size-200'}>
            <Box
                testId={'Box-LEARNING PARAMETERS'}
                contentClassName={classes.scrollableContent}
                title={'LEARNING PARAMETERS'}
                content={<TrainingConfigurationParametersList parameters={learningParameters} />}
            />
            <Box
                testId={'Box-FILTERS'}
                contentClassName={classes.scrollableContent}
                title={'FILTERS'}
                content={<TrainingConfigurationParametersList parameters={filteringGroup?.parameters || []} />}
            />
            <Box
                testId={'Box-AUGMENTATIONS'}
                contentClassName={classes.scrollableContent}
                title={'AUGMENTATIONS'}
                content={<TrainingConfigurationParametersList parameters={augmentationGroup?.parameters || []} />}
            />
        </Grid>
    );
};
