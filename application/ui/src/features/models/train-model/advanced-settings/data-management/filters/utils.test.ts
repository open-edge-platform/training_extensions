// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    getMockedConfigurationParameter,
    getMockedConfigurationParameterGroup,
} from 'mocks/mock-training-configuration';

import { checkIfFiltersAreEnabled, FilterConfigurableParameterGroup } from './utils';

const buildFilterGroup = (enableValue: boolean, value = 10): FilterConfigurableParameterGroup => {
    return getMockedConfigurationParameterGroup({
        parameters: [
            getMockedConfigurationParameter({ value_type: 'bool', key: 'enable', value: enableValue }),
            getMockedConfigurationParameter({ value_type: 'int', key: 'value', value }),
        ],
    }) as FilterConfigurableParameterGroup;
};

describe('checkIfFiltersAreEnabled', () => {
    it('returns false when given an empty array', () => {
        expect(checkIfFiltersAreEnabled([])).toBe(false);
    });

    it('returns false when all enable parameters are false', () => {
        const groups = [buildFilterGroup(false), buildFilterGroup(false)];
        expect(checkIfFiltersAreEnabled(groups)).toBe(false);
    });

    it('returns true when at least one enable parameter is true', () => {
        const groups = [buildFilterGroup(false), buildFilterGroup(true)];
        expect(checkIfFiltersAreEnabled(groups)).toBe(true);
    });

    it('returns true when all enable parameters are true', () => {
        const groups = [buildFilterGroup(true), buildFilterGroup(true)];
        expect(checkIfFiltersAreEnabled(groups)).toBe(true);
    });

    it('returns true for a single group with enable set to true', () => {
        const groups = [buildFilterGroup(true)];
        expect(checkIfFiltersAreEnabled(groups)).toBe(true);
    });

    it('returns false for a single group with enable set to false', () => {
        const groups = [buildFilterGroup(false)];
        expect(checkIfFiltersAreEnabled(groups)).toBe(false);
    });

    it('ignores non-enable bool parameters', () => {
        const groupWithNonEnableBool = getMockedConfigurationParameterGroup({
            parameters: [
                getMockedConfigurationParameter({ value_type: 'bool', key: 'some_other_bool', value: true }),
                getMockedConfigurationParameter({ value_type: 'int', key: 'threshold', value: 5 }),
            ],
        }) as FilterConfigurableParameterGroup;

        expect(checkIfFiltersAreEnabled([groupWithNonEnableBool])).toBe(false);
    });
});
