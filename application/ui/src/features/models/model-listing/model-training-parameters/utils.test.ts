// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedTrainingConfiguration } from 'mocks/mock-training-configuration';

import { TrainingConfigurationParameter } from '../../../../constants/shared-types';
import { findGroupByKey, flattenParameters, isParameterGroup } from './utils';

describe('Training parameters utils', () => {
    it('finds a nested top-level group by key', () => {
        const parameters = getMockedTrainingConfiguration();

        const group = findGroupByKey(parameters, 'dataset_preparation');

        expect(group?.name).toBe('Dataset preparation');
    });

    it('returns undefined when group key does not exist', () => {
        const parameters = getMockedTrainingConfiguration();

        expect(findGroupByKey(parameters, 'non_existing_key')).toBeUndefined();
    });

    it('identifies parameter groups correctly', () => {
        const parameters = getMockedTrainingConfiguration();
        const trainingGroup = findGroupByKey(parameters, 'training');

        expect(isParameterGroup(parameters[0])).toBe(true);
        expect(isParameterGroup(trainingGroup?.parameters[0] as TrainingConfigurationParameter)).toBe(false);
    });

    it('flattens nested groups and formats values', () => {
        const parameters = getMockedTrainingConfiguration();
        const datasetPreparationGroup = findGroupByKey(parameters, 'dataset_preparation');
        const augmentationGroup = findGroupByKey(datasetPreparationGroup?.parameters, 'augmentation');

        const rows = flattenParameters(augmentationGroup?.parameters);

        expect(rows).toEqual([
            { name: 'Mosaic / Enable', value: 'On' },
            { name: 'Gaussian blur / Sigma range', value: '0.1 - 2' },
            { name: 'Gaussian blur / Probability', value: '0.5' },
        ]);
    });

    it('returns an empty list when parameters are missing', () => {
        expect(flattenParameters(undefined)).toEqual([]);
    });
});
