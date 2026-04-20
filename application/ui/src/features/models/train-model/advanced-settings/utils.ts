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

/**
 * Filters a list of training configuration parameters based on their `depends_on` conditions.
 *
 * For each parameter that declares a `depends_on` mapping, this function looks for the
 * referenced parameter at the same depth. A parameter is included in the output only if:
 * - It has no `depends_on` declaration, OR
 * - Its dependency parameter cannot be found (or is a group, not a leaf parameter), OR
 * - The dependency parameter's current value satisfies the condition specified in `depends_on`
 *   (either an exact match or inclusion in an allowed-values array).
 *
 * Parameter groups without a `depends_on` are recursively processed so that their nested
 * parameters are also filtered.
 *
 * @param parameters - The flat or nested list of {@link TrainingConfigurationParameter} objects to filter.
 * @returns A new array containing only the parameters (and recursively filtered groups) that
 *          satisfy their dependency conditions.
 */
export const filterDependentParameters = (
    parameters: TrainingConfigurationParameter[]
): TrainingConfigurationParameter[] => {
    return parameters.reduce<TrainingConfigurationParameter[]>((acc, curr, _idx, parametersOnTheSameDepth) => {
        if (curr.depends_on != null) {
            const dependentParameter = parametersOnTheSameDepth.find(
                (parameter) => curr.depends_on != null && curr.depends_on[parameter.key] != null
            );

            if (dependentParameter === undefined || !isParameter(dependentParameter)) {
                acc.push(curr);
                return acc;
            }

            const dependsOnValue = curr.depends_on[dependentParameter.key];
            const shouldBeVisible = Array.isArray(dependsOnValue)
                ? dependsOnValue.includes(dependentParameter.value)
                : dependsOnValue === dependentParameter.value;

            if (shouldBeVisible) {
                acc.push(curr);
            }

            return acc;
        }

        if (isParameterGroup(curr)) {
            const groupedParameters = filterDependentParameters(curr.parameters);

            acc.push({ ...curr, parameters: groupedParameters });

            return acc;
        }

        acc.push(curr);

        return acc;
    }, []);
};
