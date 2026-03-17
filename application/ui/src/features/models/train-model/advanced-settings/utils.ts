// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { get, isBoolean, isNumber, isObject } from 'lodash-es';

import {
    BoolConfigurableParameter,
    ConfigurableParameter,
    EnumConfigurableParameter,
    NumberConfigurableParameter,
    NumberEnumConfigurableParameter,
    StringConfigurableParameter,
    type ConfigurableParameterGroup,
    type TrainingConfigurationParameter,
} from '../../../../constants/shared-types';
import { isParameterGroup } from '../../model-listing/model-training-parameters/utils';

const getDecimalPoints = (value: number): number => {
    return Math.abs(Math.ceil(Math.log10(value)));
};

export const getFloatingPointStep = (minValue: number, maxValue: number): number => {
    const exponent = getDecimalPoints(maxValue - minValue);

    return 1 / Math.pow(10, exponent + 3);
};

export const isBoolEnableParameter = (parameter: ConfigurableParameter) => {
    return parameter.value_type === 'bool' && parameter.key === 'enable';
};

export const isBoolParameter = (input: unknown): input is BoolConfigurableParameter => {
    return isObject(input) && get(input, 'value_type') === 'bool' && isBoolean(get(input, 'value'));
};

export const isStringParameter = (input: ConfigurableParameter): input is StringConfigurableParameter => {
    return input.value_type === 'str';
};

export const isNumberParameter = (input: unknown): input is NumberConfigurableParameter => {
    return (
        isObject(input) &&
        (get(input, 'value_type') === 'float' || get(input, 'value_type') === 'int') &&
        isNumber(get(input, 'value'))
    );
};

export const isEnumNumberParameter = (input: ConfigurableParameter): input is NumberEnumConfigurableParameter => {
    return isNumberParameter(input) && input.allowed_values != null;
};

export const isEnumStringParameter = (input: ConfigurableParameter): input is StringConfigurableParameter => {
    return isStringParameter(input) && input.allowed_values != null;
};

export const isEnumParameter = (input: ConfigurableParameter): input is EnumConfigurableParameter => {
    return isEnumNumberParameter(input) || isEnumStringParameter(input);
};

export const isConfigurationParameter = (input: unknown): input is ConfigurableParameter => {
    return isObject(input) && 'key' in input && 'name' in input && 'description' in input;
};

export const replaceByKey = (
    parameters: TrainingConfigurationParameter[],
    key: string,
    replace: (match: ConfigurableParameterGroup) => ConfigurableParameterGroup
): TrainingConfigurationParameter[] =>
    parameters.map((parameter) => {
        if (isParameterGroup(parameter) && parameter.key === key) {
            return replace(parameter);
        }
        return parameter;
    });
