// Copyright (C) 2022-2025 Intel Corporation
// LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import { getMockedConfigurationParameter } from '../../../../../../../../test-utils/mocked-items-factory/mocked-configuration-parameters';
import { areSubsetsSizesValid, getSubsetsSizes } from './utils';

describe('getSubsetsSizes', () => {
    const datasetSizeParameter = [
        getMockedConfigurationParameter({
            type: 'int',
            value: 101,
            key: 'dataset_size',
        }),
    ];

    it('calculates subsets sizes for equal distribution', () => {
        const validationRatio = 33;
        const testRatio = 33;

        const result = getSubsetsSizes(datasetSizeParameter, validationRatio, testRatio);

        expect(result.trainingSubsetSize).toBe(35);
        expect(result.validationSubsetSize).toBe(33);
        expect(result.testSubsetSize).toBe(33);
    });

    it('returns correct sizes for zero validation and test ratios', () => {
        const validationRatio = 0;
        const testRatio = 0;

        const result = getSubsetsSizes(datasetSizeParameter, validationRatio, testRatio);

        expect(result.trainingSubsetSize).toBe(101);
        expect(result.validationSubsetSize).toBe(0);
        expect(result.testSubsetSize).toBe(0);
    });

    it('handles maximum test ratio correctly', () => {
        const validationRatio = 0;
        const testRatio = 100;

        const result = getSubsetsSizes(datasetSizeParameter, validationRatio, testRatio);

        expect(result.trainingSubsetSize).toBe(0);
        expect(result.validationSubsetSize).toBe(0);
        expect(result.testSubsetSize).toBe(101);
    });

    it('handles maximum validation ratio correctly', () => {
        const validationRatio = 100;
        const testRatio = 0;

        const result = getSubsetsSizes(datasetSizeParameter, validationRatio, testRatio);

        expect(result.trainingSubsetSize).toBe(0);
        expect(result.validationSubsetSize).toBe(101);
        expect(result.testSubsetSize).toBe(0);
    });

    it('returns zero sizes for zero dataset size', () => {
        const zeroDatasetParameters = [
            getMockedConfigurationParameter({
                type: 'int',
                value: 0,
                key: 'dataset_size',
            }),
        ];

        const validationRatio = 50;
        const testRatio = 50;

        const result = getSubsetsSizes(zeroDatasetParameters, validationRatio, testRatio);

        expect(result.trainingSubsetSize).toBe(0);
        expect(result.validationSubsetSize).toBe(0);
        expect(result.testSubsetSize).toBe(0);
    });

    it('returns zero sizes when dataset size parameter is missing', () => {
        const validationRatio = 50;
        const testRatio = 50;

        const result = getSubsetsSizes([], validationRatio, testRatio);

        expect(result.trainingSubsetSize).toBe(0);
        expect(result.validationSubsetSize).toBe(0);
        expect(result.testSubsetSize).toBe(0);
    });
});

describe('areSubsetsSizesValid', () => {
    it('returns true for valid non-zero subset sizes', () => {
        const params = [
            getMockedConfigurationParameter({
                key: 'training',
                type: 'int',
                name: 'Training percentage',
                value: 70,
                description: 'Percentage of data to use for training',
                defaultValue: 70,
                maxValue: 100,
                minValue: 1,
            }),
            getMockedConfigurationParameter({
                key: 'validation',
                type: 'int',
                name: 'Validation percentage',
                value: 20,
                description: 'Percentage of data to use for validation',
                defaultValue: 20,
                maxValue: 100,
                minValue: 1,
            }),
            getMockedConfigurationParameter({
                key: 'test',
                type: 'int',
                name: 'Test percentage',
                value: 10,
                description: 'Percentage of data to use for testing',
                defaultValue: 10,
                maxValue: 100,
                minValue: 1,
            }),
            getMockedConfigurationParameter({
                type: 'int',
                value: 100,
                key: 'dataset_size',
            }),
        ];
        expect(areSubsetsSizesValid(params)).toBe(true);
    });

    it('returns false if dataset size is zero', () => {
        const params = [
            getMockedConfigurationParameter({
                key: 'training',
                type: 'int',
                name: 'Training percentage',
                value: 70,
                description: 'Percentage of data to use for training',
                defaultValue: 70,
                maxValue: 100,
                minValue: 1,
            }),
            getMockedConfigurationParameter({
                key: 'validation',
                type: 'int',
                name: 'Validation percentage',
                value: 20,
                description: 'Percentage of data to use for validation',
                defaultValue: 20,
                maxValue: 100,
                minValue: 1,
            }),
            getMockedConfigurationParameter({
                key: 'test',
                type: 'int',
                name: 'Test percentage',
                value: 10,
                description: 'Percentage of data to use for testing',
                defaultValue: 10,
                maxValue: 100,
                minValue: 1,
            }),
            getMockedConfigurationParameter({
                type: 'int',
                value: 0,
                key: 'dataset_size',
            }),
        ];
        expect(areSubsetsSizesValid(params)).toBe(false);
    });

    it('returns false if dataset size parameter is missing', () => {
        expect(
            areSubsetsSizesValid([
                getMockedConfigurationParameter({
                    key: 'training',
                    type: 'int',
                    name: 'Training percentage',
                    value: 70,
                    description: 'Percentage of data to use for training',
                    defaultValue: 70,
                    maxValue: 100,
                    minValue: 1,
                }),
                getMockedConfigurationParameter({
                    key: 'validation',
                    type: 'int',
                    name: 'Validation percentage',
                    value: 20,
                    description: 'Percentage of data to use for validation',
                    defaultValue: 20,
                    maxValue: 100,
                    minValue: 1,
                }),
                getMockedConfigurationParameter({
                    key: 'test',
                    type: 'int',
                    name: 'Test percentage',
                    value: 10,
                    description: 'Percentage of data to use for testing',
                    defaultValue: 10,
                    maxValue: 100,
                    minValue: 1,
                }),
            ])
        ).toBe(false);
    });

    it('returns false if any subset size is zero', () => {
        const params = [
            getMockedConfigurationParameter({
                key: 'training',
                type: 'int',
                name: 'Training percentage',
                value: 0,
                description: 'Percentage of data to use for training',
                defaultValue: 70,
                maxValue: 100,
                minValue: 1,
            }),
            getMockedConfigurationParameter({
                key: 'validation',
                type: 'int',
                name: 'Validation percentage',
                value: 20,
                description: 'Percentage of data to use for validation',
                defaultValue: 20,
                maxValue: 100,
                minValue: 1,
            }),
            getMockedConfigurationParameter({
                key: 'test',
                type: 'int',
                name: 'Test percentage',
                value: 10,
                description: 'Percentage of data to use for testing',
                defaultValue: 10,
                maxValue: 100,
                minValue: 1,
            }),
            getMockedConfigurationParameter({
                type: 'int',
                value: 6,
                key: 'dataset_size',
            }),
        ];
        expect(areSubsetsSizesValid(params)).toBe(false);
    });
});
