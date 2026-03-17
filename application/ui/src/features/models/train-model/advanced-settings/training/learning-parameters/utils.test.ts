// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    getMockedConfigurationParameter,
    getMockedConfigurationParameterGroup,
} from 'mocks/mock-training-configuration';

import { TrainingConfigurationParameter } from '../../../../../../constants/shared-types';
import { isInputSizeHeightParameter, isInputSizeWidthParameter } from './utils';

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
