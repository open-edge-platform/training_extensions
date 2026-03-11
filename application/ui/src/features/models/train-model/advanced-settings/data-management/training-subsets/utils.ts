// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { NumberConfigurableParameter, TrainingConfiguration } from '../../../../../../constants/shared-types';
import { findGroupByKey } from '../../../../model-listing/model-training-parameters/utils';

export type SubsetSplitParameters = [
    NumberConfigurableParameter,
    NumberConfigurableParameter,
    NumberConfigurableParameter,
];

export const MAX_RATIO_VALUE = 100;

export const TEST_SUBSET_KEY = 'test';
export const VALIDATION_SUBSET_KEY = 'validation';
export const TRAINING_SUBSET_KEY = 'training';

export const getSubsets = (subsetsParameters: SubsetSplitParameters) => {
    const validationSubset = subsetsParameters.find(
        (parameter) => parameter.key === VALIDATION_SUBSET_KEY
    ) as NumberConfigurableParameter;
    const trainingSubset = subsetsParameters.find(
        (parameter) => parameter.key === TRAINING_SUBSET_KEY
    ) as NumberConfigurableParameter;
    const testSubset = subsetsParameters.find(
        (parameter) => parameter.key === TEST_SUBSET_KEY
    ) as NumberConfigurableParameter;

    return {
        trainingSubset,
        validationSubset,
        testSubset,
    };
};

export const getSubsetSplitParameters = (trainingConfiguration: TrainingConfiguration) => {
    const datasetPreparation = findGroupByKey(trainingConfiguration.parameters, 'dataset_preparation')?.parameters;

    if (datasetPreparation === undefined) return undefined;

    const subsetSplit = findGroupByKey(datasetPreparation, 'subset_split');

    if (subsetSplit === undefined) return undefined;

    return subsetSplit.parameters as SubsetSplitParameters;
};
