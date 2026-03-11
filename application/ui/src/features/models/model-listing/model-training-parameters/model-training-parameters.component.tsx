// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Fragment } from 'react';

import { Grid, Text } from '@geti/ui';

import { TrainingConfigurationParameter } from '../../../../constants/shared-types';
import { useGetModelTrainingConfiguration } from '../../hooks/api/use-get-model-training-configuration.hook';
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
            {parameterRows.map((row) => (
                <Fragment key={`${row.name}-${row.value}`}>
                    <Text>{row.name}</Text>
                    <Text>{row.value}</Text>
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

    return (
        <Grid columns={['1fr', '1fr', '1fr']} gap={'size-200'}>
            <Box
                customClasses={classes.box}
                title={'LEARNING PARAMETERS'}
                content={<TrainingConfigurationParametersList parameters={trainingGroup?.parameters || []} />}
            />
            <Box
                customClasses={classes.box}
                title={'FILTERS'}
                content={<TrainingConfigurationParametersList parameters={filteringGroup?.parameters || []} />}
            />
            <Box
                customClasses={classes.box}
                title={'AUGMENTATIONS'}
                content={<TrainingConfigurationParametersList parameters={augmentationGroup?.parameters || []} />}
            />
        </Grid>
    );
};
