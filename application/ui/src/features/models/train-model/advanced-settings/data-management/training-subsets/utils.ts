// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    ConfigurableParameter,
    NumberConfigurableParameter,
    TrainingConfiguration,
} from '../../../../../../constants/shared-types';
import { findGroupByKey } from '../../../../model-listing/model-training-parameters/utils';
import { isNumberParameter } from '../../utils';

export type SubsetSplitParameters = ConfigurableParameter[];

const getDatasetSize = (subsetSplitParameters: SubsetSplitParameters): number => {
    const datasetSize = subsetSplitParameters.find((parameter) => parameter.key === 'dataset_size');

    if (isNumberParameter(datasetSize)) {
        return datasetSize.value;
    }

    return 0;
};

export const getSubsetsSizes = (
    subsetSplitParameters: SubsetSplitParameters,
    validationSubsetRatio: number,
    testSubsetRatio: number
) => {
    const datasetSize = getDatasetSize(subsetSplitParameters);

    const validationSubsetSize = Math.floor(datasetSize * (validationSubsetRatio / 100));
    const testSubsetSize = Math.floor(datasetSize * (testSubsetRatio / 100));
    const trainingSubsetSize = datasetSize - validationSubsetSize - testSubsetSize;

    return {
        trainingSubsetSize,
        validationSubsetSize,
        testSubsetSize,
    };
};

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

export const areSubsetsSizesValid = (subsetParameters: SubsetSplitParameters): boolean => {
    const { validationSubset, testSubset } = getSubsets(subsetParameters);

    const newSubsetSizes = getSubsetsSizes(subsetParameters, validationSubset.value, testSubset.value);

    return ![
        newSubsetSizes.trainingSubsetSize,
        newSubsetSizes.validationSubsetSize,
        newSubsetSizes.testSubsetSize,
    ].some((size) => size === 0);
};

export const getSubsetSplitParameters = (trainingConfiguration: TrainingConfiguration) => {
    const datasetPreparation = findGroupByKey(trainingConfiguration.parameters, 'dataset_preparation')?.parameters;

    return findGroupByKey(datasetPreparation, 'subset_split')?.parameters as NumberConfigurableParameter[];
};
