// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    getMockedConfigurationParameter,
    getMockedConfigurationParameterGroup,
} from 'mocks/mock-training-configuration';

import { TrainingConfigurationParameter } from '../../../../../../constants/shared-types';
import { isParameter, isParameterGroup } from '../../../../model-listing/model-training-parameters/utils';
import { learningParameters } from './mocks';
import {
    filterDependentParameters,
    isInputSizeHeightParameter,
    isInputSizeWidthParameter,
    LearningConfigurationGroup,
} from './utils';

const buildEnumNumberParameter = (key: string): TrainingConfigurationParameter =>
    getMockedConfigurationParameter({ value_type: 'int', key, allowed_values: [320, 416, 512, 640] });

describe('isInputSizeWidthParameter', () => {
    it('returns true for a number enum parameter with key "input_size_width"', () => {
        const parameter = buildEnumNumberParameter('input_size_width');

        expect(isInputSizeWidthParameter(parameter)).toBe(true);
    });

    it('returns false for a number enum parameter with a different key', () => {
        const parameter = buildEnumNumberParameter('input_size_height');

        expect(isInputSizeWidthParameter(parameter)).toBe(false);
    });

    it('returns false for a number parameter without allowed_values', () => {
        const parameter = getMockedConfigurationParameter({
            value_type: 'int',
            key: 'input_size_width',
            allowed_values: null,
        });

        expect(isInputSizeWidthParameter(parameter)).toBe(false);
    });

    it('returns false for a bool parameter with key "input_size_width"', () => {
        const parameter = getMockedConfigurationParameter({ value_type: 'bool', key: 'input_size_width' });

        expect(isInputSizeWidthParameter(parameter)).toBe(false);
    });

    it('returns false for a parameter group', () => {
        const group = getMockedConfigurationParameterGroup({ key: 'input_size_width' });

        expect(isInputSizeWidthParameter(group)).toBe(false);
    });
});

describe('isInputSizeHeightParameter', () => {
    it('returns true for a number enum parameter with key "input_size_height"', () => {
        const parameter = buildEnumNumberParameter('input_size_height');

        expect(isInputSizeHeightParameter(parameter)).toBe(true);
    });

    it('returns false for a number enum parameter with a different key', () => {
        const parameter = buildEnumNumberParameter('input_size_width');

        expect(isInputSizeHeightParameter(parameter)).toBe(false);
    });

    it('returns false for a number parameter without allowed_values', () => {
        const parameter = getMockedConfigurationParameter({
            value_type: 'int',
            key: 'input_size_height',
            allowed_values: null,
        });

        expect(isInputSizeHeightParameter(parameter)).toBe(false);
    });

    it('returns false for a bool parameter with key "input_size_height"', () => {
        const parameter = getMockedConfigurationParameter({ value_type: 'bool', key: 'input_size_height' });

        expect(isInputSizeHeightParameter(parameter)).toBe(false);
    });

    it('returns false for a parameter group', () => {
        const group = getMockedConfigurationParameterGroup({ key: 'input_size_height' });

        expect(isInputSizeHeightParameter(group)).toBe(false);
    });
});

describe('filterDependentParameters', () => {
    it('keeps independent parameters in their original order', () => {
        const result = filterDependentParameters(learningParameters.parameters);
        const keys = result.filter(isParameter).map((p) => p.key);

        expect(keys).toContain('max_epochs');
        expect(keys).toContain('batch_size');
        expect(keys.indexOf('max_epochs')).toBeLessThan(keys.indexOf('batch_size'));
    });

    it('places dependent parameters immediately after the parameter they depend on', () => {
        const schedulerGroup = learningParameters.parameters.find(
            (p) => isParameterGroup(p) && p.key === 'scheduler'
        ) as LearningConfigurationGroup;
        const result = filterDependentParameters(schedulerGroup.parameters);
        const keys = result.filter(isParameter).map((p) => p.key);

        const typeIndex = keys.indexOf('type');
        const factorIndex = keys.indexOf('factor');
        const patienceIndex = keys.indexOf('patience');

        // 'factor' and 'patience' depend on 'type' === 'reduce_lr_on_plateau' (which is the current value)
        // so they should immediately follow 'type'
        expect(factorIndex).toBe(typeIndex + 1);
        expect(patienceIndex).toBe(typeIndex + 2);
    });

    it('excludes dependent parameters whose condition does not match the current value', () => {
        const schedulerGroup = learningParameters.parameters.find(
            (p) => isParameterGroup(p) && p.key === 'scheduler'
        ) as LearningConfigurationGroup;
        const result = filterDependentParameters(schedulerGroup.parameters);
        const keys = result.filter(isParameter).map((p) => p.key);

        // 'min_lr' depends on type === 'cosine_annealing', but type === 'reduce_lr_on_plateau'
        expect(keys).not.toContain('min_lr');
    });

    it('preserves parameter groups in the output', () => {
        const result = filterDependentParameters(learningParameters.parameters);
        const groupKeys = result.filter(isParameterGroup).map((p) => p.key);

        expect(groupKeys).toContain('early_stopping');
        expect(groupKeys).toContain('scheduler');
        expect(groupKeys).toContain('gradient_accumulation');
        expect(groupKeys).toContain('gradient_clip');
    });

    it('returns an empty array when given an empty array', () => {
        expect(filterDependentParameters([])).toEqual([]);
    });

    it('returns a single independent parameter unchanged', () => {
        const param = getMockedConfigurationParameter({ value_type: 'int', key: 'solo', depends_on: null });
        const result = filterDependentParameters([param]);

        expect(result).toEqual([param]);
    });

    it('does not include standalone dependent parameters that have no matching parent', () => {
        const dependent = getMockedConfigurationParameter({
            value_type: 'float',
            key: 'orphan',
            depends_on: { some_key: 'some_value' },
        });
        const result = filterDependentParameters([dependent]);

        expect(result).toEqual([]);
    });
});
