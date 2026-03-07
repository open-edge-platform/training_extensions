// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Grid, Text, View } from '@geti/ui';

import { ImportDatasetAsNewProjectState } from '../../../../dataset/import-export/import-dataset/util';

import classes from './progress-stepper.module.scss';

type ProgressStepperProps = {
    currentStep: ImportDatasetAsNewProjectState;
};

const isStepOne = (step: ImportDatasetAsNewProjectState) => ['uploading', 'preparing'].includes(step);
const isStepOneCompleted = (step: ImportDatasetAsNewProjectState) =>
    ['taskTypeSelection', 'labelMapping'].includes(step);

const isStepTwo = (step: ImportDatasetAsNewProjectState) => ['taskTypeSelection'].includes(step);
const isStepTwoCompleted = (step: ImportDatasetAsNewProjectState) => ['labelMapping'].includes(step);

const isStepThree = (step: ImportDatasetAsNewProjectState) => ['labelMapping'].includes(step);

export const ProgressStepper = ({ currentStep }: ProgressStepperProps) => {
    return (
        <Grid
            gap={'size-100'}
            width={'100%'}
            maxWidth={'35rem'}
            alignItems={'center'}
            justifyItems={'center'}
            columns={['1.875rem', '1fr', '1.875rem', '1fr', '1.875rem']}
        >
            <div
                aria-label='step one'
                className={classes.step}
                data-active={isStepOne(currentStep)}
                data-completed={isStepOneCompleted(currentStep)}
            >
                {isStepOneCompleted(currentStep) ? '✓' : '1'}
            </div>
            <View UNSAFE_className={classes.stepDivider}></View>
            <div
                aria-label='step two'
                className={classes.step}
                data-active={isStepTwo(currentStep)}
                data-completed={isStepTwoCompleted(currentStep)}
            >
                {isStepTwoCompleted(currentStep) ? '✓' : '2'}
            </div>
            <View UNSAFE_className={classes.stepDivider}></View>
            <div aria-label='step three' className={classes.step} data-active={isStepThree(currentStep)}>
                3
            </div>

            <Text>Dataset</Text>
            <View></View>
            <Text UNSAFE_style={{ width: dimensionValue('size-800'), textAlign: 'center' }}>Task type</Text>
            <View></View>
            <Text>Label</Text>
        </Grid>
    );
};
