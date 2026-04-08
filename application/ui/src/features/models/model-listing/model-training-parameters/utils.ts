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

export const isParameter = (
    parameter: TrainingConfigurationParameter | undefined
): parameter is ConfigurableParameter => {
    return parameter?.type === 'parameter';
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
 *   { name: "Data augmentation:", value: "", depth: 0, isGroup: true },
 *   { name: "Random affine:", value: "", depth: 1, isGroup: true },
 *   { name: "Rotation degrees", value: "10", depth: 2, isGroup: false },
 *   { name: "Scaling ratio range", value: "0.5 - 1.5", depth: 2, isGroup: false }
 * ]
 *
 */

type FlattenedParameterRow = {
    name: string;
    value: string;
    depth: number;
    isGroup: boolean;
};

export const flattenParameters = (
    parameters: TrainingConfigurationParameter[] | undefined,
    depth = 0
): FlattenedParameterRow[] => {
    if (!parameters) {
        return [];
    }

    return parameters.flatMap((parameter) => {
        if (isParameterGroup(parameter)) {
            const groupRow: FlattenedParameterRow = {
                name: `${parameter.name}:`,
                value: '',
                depth,
                isGroup: true,
            };

            return [groupRow, ...flattenParameters(parameter.parameters, depth + 1)];
        }

        return [
            {
                name: parameter.name,
                value: formatParameterValue(parameter.value),
                depth,
                isGroup: false,
            },
        ];
    });
};
