// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { get, isBoolean, isNumber, isObject } from 'lodash-es';

import {
    BoolParameter,
    ConfigurationParameter,
    EnumConfigurationParameter,
    NumberParameter,
} from '../../configuration.interface';

const getDecimalPoints = (value: number): number => {
    return Math.abs(Math.ceil(Math.log10(value)));
};

export const getFloatingPointStep = (minValue: number, maxValue: number): number => {
    const exponent = getDecimalPoints(maxValue - minValue);

    return 1 / Math.pow(10, exponent + 3);
};

export const isBoolEnableParameter = (parameter: ConfigurationParameter) => {
    return parameter.type === 'bool' && parameter.key === 'enable';
};

export const isBoolParameter = (input: unknown): input is BoolParameter => {
    return isObject(input) && get(input, 'type') === 'bool' && isBoolean(get(input, 'value'));
};

export const isEnumParameter = (input: unknown): input is EnumConfigurationParameter => {
    return (
        isObject(input) &&
        get(input, 'type') === 'enum' &&
        get(input, 'allowed_values') !== undefined &&
        get(input, 'value') !== undefined
    );
};

export const isNumberParameter = (input: unknown): input is NumberParameter => {
    return (
        isObject(input) &&
        (get(input, 'type') === 'float' || get(input, 'type') === 'int') &&
        isNumber(get(input, 'value'))
    );
};

export const isConfigurationParameter = (input: unknown): input is ConfigurationParameter => {
    return isObject(input) && 'key' in input && 'name' in input && 'description' in input;
};
