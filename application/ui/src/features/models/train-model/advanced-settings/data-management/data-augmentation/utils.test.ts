// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    getMockedConfigurationParameter,
    getMockedConfigurationParameterGroup,
} from 'mocks/mock-training-configuration';

import { DataAugmentationConfigurationParameters, isDataAugmentationEnabled } from './utils';

const buildAugmentationGroup = (enableValue: boolean) => {
    return getMockedConfigurationParameterGroup({
        parameters: [
            getMockedConfigurationParameter({ value_type: 'bool', key: 'enable', value: enableValue }),
            getMockedConfigurationParameter({ value_type: 'float', key: 'probability', value: 0.5 }),
        ],
    });
};

const buildAugmentationGroupWithoutEnable = () => {
    return getMockedConfigurationParameterGroup({
        parameters: [getMockedConfigurationParameter({ value_type: 'float', key: 'probability', value: 0.5 })],
    });
};

const buildDataAugmentationConfigurableParameters = (groups: DataAugmentationConfigurationParameters[]) => {
    return getMockedConfigurationParameterGroup({
        parameters: groups,
    });
};

describe('isDataAugmentationEnabled', () => {
    it('returns false when there are no augmentation groups', () => {
        const params = buildDataAugmentationConfigurableParameters([]);
        expect(isDataAugmentationEnabled(params)).toBe(false);
    });

    it('returns false when all enable parameters are false', () => {
        const params = buildDataAugmentationConfigurableParameters([
            buildAugmentationGroup(false),
            buildAugmentationGroup(false),
        ]);
        expect(isDataAugmentationEnabled(params)).toBe(false);
    });

    it('returns true when at least one enable parameter is true', () => {
        const params = buildDataAugmentationConfigurableParameters([
            buildAugmentationGroup(false),
            buildAugmentationGroup(true),
        ]);
        expect(isDataAugmentationEnabled(params)).toBe(true);
    });

    it('returns true when all enable parameters are true', () => {
        const params = buildDataAugmentationConfigurableParameters([
            buildAugmentationGroup(true),
            buildAugmentationGroup(true),
        ]);
        expect(isDataAugmentationEnabled(params)).toBe(true);
    });

    it('returns true for a single group with enable set to true', () => {
        const params = buildDataAugmentationConfigurableParameters([buildAugmentationGroup(true)]);
        expect(isDataAugmentationEnabled(params)).toBe(true);
    });

    it('returns false for a single group with enable set to false', () => {
        const params = buildDataAugmentationConfigurableParameters([buildAugmentationGroup(false)]);
        expect(isDataAugmentationEnabled(params)).toBe(false);
    });

    it('returns false when no group has an enable parameter', () => {
        const params = buildDataAugmentationConfigurableParameters([
            buildAugmentationGroupWithoutEnable(),
            buildAugmentationGroupWithoutEnable(),
        ]);
        expect(isDataAugmentationEnabled(params)).toBe(false);
    });

    it('ignores non-enable bool parameters', () => {
        const groupWithNonEnableBool = getMockedConfigurationParameterGroup({
            parameters: [
                getMockedConfigurationParameter({ value_type: 'bool', key: 'some_other_bool', value: true }),
                getMockedConfigurationParameter({ value_type: 'float', key: 'probability', value: 0.5 }),
            ],
        }) as DataAugmentationConfigurationParameters;

        const params = buildDataAugmentationConfigurableParameters([groupWithNonEnableBool]);
        expect(isDataAugmentationEnabled(params)).toBe(false);
    });

    it('returns true when deim_framework parameter is set to true', () => {
        const groupWithDeimFramework = getMockedConfigurationParameterGroup({
            parameters: [
                getMockedConfigurationParameter({ value_type: 'bool', key: 'deim_framework', value: true }),
                getMockedConfigurationParameter({ value_type: 'float', key: 'probability', value: 0.5 }),
                buildAugmentationGroup(true),
            ],
        });

        expect(isDataAugmentationEnabled(groupWithDeimFramework)).toBe(true);
    });

    it('returns true when deim_framework parameter is set to false and at least one enable parameter is true', () => {
        const groupWithDeimFramework = getMockedConfigurationParameterGroup({
            parameters: [
                getMockedConfigurationParameter({ value_type: 'bool', key: 'deim_framework', value: false }),
                getMockedConfigurationParameter({ value_type: 'float', key: 'probability', value: 0.5 }),
                buildAugmentationGroup(true),
            ],
        });

        expect(isDataAugmentationEnabled(groupWithDeimFramework)).toBe(true);
    });

    it('returns false when deim_framework parameter is set to false and at least one enable parameter is false', () => {
        const groupWithDeimFramework = getMockedConfigurationParameterGroup({
            parameters: [
                getMockedConfigurationParameter({ value_type: 'bool', key: 'deim_framework', value: false }),
                getMockedConfigurationParameter({ value_type: 'float', key: 'probability', value: 0.5 }),
                buildAugmentationGroup(false),
            ],
        });

        expect(isDataAugmentationEnabled(groupWithDeimFramework)).toBe(false);
    });
});
