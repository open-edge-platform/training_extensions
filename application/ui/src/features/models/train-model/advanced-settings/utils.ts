// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { get, isBoolean, isNumber, isObject } from 'lodash-es';

import type {
    BoolConfigurableParameter,
    ConfigurableParameter,
    ConfigurableParameterGroup,
    NumberConfigurableParameter,
    NumberEnumConfigurableParameter,
    StringConfigurableParameter,
    StringEnumConfigurableParameter,
    TrainingConfigurationParameter,
} from '../../../../constants/shared-types';
import { isParameter, isParameterGroup } from '../../model-listing/model-training-parameters/utils';

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

export const isEnumStringParameter = (input: ConfigurableParameter): input is StringEnumConfigurableParameter => {
    return isStringParameter(input) && input.allowed_values != null;
};

export const deepReplaceParameters = (
    parameters: TrainingConfigurationParameter[],
    updatedParameters: ConfigurableParameter[],
    targetGroupKeys?: string[],
    currentGroupKeys: string[] = []
): TrainingConfigurationParameter[] => {
    if (updatedParameters.length === 0) {
        return parameters;
    }

    return parameters.map((parameter) => {
        if (isParameterGroup(parameter)) {
            return {
                ...parameter,
                parameters: deepReplaceParameters(parameter.parameters, updatedParameters, targetGroupKeys, [
                    ...currentGroupKeys,
                    parameter.key,
                ]),
            };
        }

        if (targetGroupKeys !== undefined && targetGroupKeys.length > 0) {
            const keysMatch =
                targetGroupKeys.length === currentGroupKeys.length &&
                targetGroupKeys.every((key, index) => key === currentGroupKeys[index]);

            if (keysMatch) {
                const updatedParameter = updatedParameters.find((updatedParam) => updatedParam.key === parameter.key);

                return updatedParameter ?? parameter;
            }

            return parameter;
        }

        const updatedParameter = updatedParameters.find((updatedParam) => updatedParam.key === parameter.key);

        return updatedParameter ?? parameter;
    });
};

export type ParametersEnableGroupParameters = ConfigurableParameterGroup & {
    parameters: [BoolConfigurableParameter, ...TrainingConfigurationParameter[]];
};

export const isBoolEnableParameterGroup = (
    parameter: TrainingConfigurationParameter
): parameter is ParametersEnableGroupParameters => {
    return (
        isParameterGroup(parameter) &&
        isParameter(parameter.parameters[0]) &&
        isBoolEnableParameter(parameter.parameters[0])
    );
};

export const filterDependentParameters = (
    parameters: TrainingConfigurationParameter[]
): TrainingConfigurationParameter[] => {
    return parameters
        .reduce<TrainingConfigurationParameter[][]>((acc, curr, _idx, parametersOnTheDepth) => {
            if (curr.depends_on != null) {
                const dependentParameter = parametersOnTheDepth.find(
                    (parameter) => curr.depends_on != null && curr.depends_on[parameter.key] != null
                );

                if (dependentParameter === undefined || !isParameter(dependentParameter)) {
                    acc.push([curr]);
                    return acc;
                }

                const dependsOnValue = curr.depends_on[dependentParameter.key];
                const shouldBeVisible = Array.isArray(dependsOnValue)
                    ? dependsOnValue.includes(dependentParameter.value)
                    : dependsOnValue === dependentParameter.value;

                if (shouldBeVisible) {
                    acc.push([curr]);
                }

                return acc;
            }

            if (isParameterGroup(curr)) {
                const groupedParameters = filterDependentParameters(curr.parameters);

                acc.push([{ ...curr, parameters: groupedParameters }]);

                return acc;
            }

            acc.push([curr]);

            return acc;
        }, [])
        .flatMap((group) => group);
};
