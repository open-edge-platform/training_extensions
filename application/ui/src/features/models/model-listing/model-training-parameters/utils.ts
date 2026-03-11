// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    ConfigurableParameter,
    ConfigurableParameterGroup,
    TrainingConfigurationParameter,
} from '../../../../constants/shared-types';

export const isParameterGroup = (
    parameter: TrainingConfigurationParameter
): parameter is ConfigurableParameterGroup => {
    return parameter.type === 'parameter_group';
};

export const isParameter = (parameter: TrainingConfigurationParameter) => {
    return parameter.type === 'parameter';
};

export const findGroupByKey = (
    parameters: TrainingConfigurationParameter[] | undefined,
    key: string
): ConfigurableParameterGroup | undefined => {
    return parameters?.find((parameter): parameter is ConfigurableParameterGroup => {
        return isParameterGroup(parameter) && parameter.key === key;
    });
};

const formatParameterValue = (value: ConfigurableParameter['value']): string => {
    if (Array.isArray(value)) {
        return value.join(' - ');
    }

    if (typeof value === 'boolean') {
        return value ? 'On' : 'Off';
    }

    if (value === null) {
        return '-';
    }

    return String(value);
};

/**
 * Flattens nested training-configuration parameters into simple display rows.
 *
 * Example input:
 * {
 *   "parameters": [
 *     {
 *       "type": "parameter_group",
 *       "key": "dataset_preparation",
 *       "name": "Dataset preparation",
 *       "parameters": [
 *         {
 *           "type": "parameter_group",
 *           "key": "augmentation",
 *           "name": "Data augmentation",
 *           "parameters": [
 *             {
 *               "type": "parameter_group",
 *               "key": "random_affine",
 *               "name": "Random affine",
 *               "parameters": [
 *                 { "type": "parameter", "name": "Rotation degrees", "value": 10.0 },
 *                 { "type": "parameter", "name": "Scaling ratio range", "value": [0.5, 1.5] }
 *               ]
 *             }
 *           ]
 *         }
 *       ]
 *     }
 *   ]
 * }
 *
 * Example output rows from this function:
 * [
 *   { name: "Data augmentation / Random affine / Rotation degrees", value: "10" },
 *   { name: "Data augmentation / Random affine / Scaling ratio range", value: "0.5 - 1.5" }
 * ]
 *
 * {ParentLabel} / {ParentLabel} / {ParameterName} {ParameterValue}
 *
 */
export const flattenParameters = (
    parameters: TrainingConfigurationParameter[] | undefined,
    parentLabel = ''
): Array<{ name: string; value: string }> => {
    if (!parameters) {
        return [];
    }

    return parameters.flatMap((parameter) => {
        if (isParameterGroup(parameter)) {
            const nextParentLabel = parentLabel ? `${parentLabel} / ${parameter.name}` : parameter.name;

            return flattenParameters(parameter.parameters, nextParentLabel);
        }

        const name = parentLabel ? `${parentLabel} / ${parameter.name}` : parameter.name;

        return [{ name, value: formatParameterValue(parameter.value) }];
    });
};
