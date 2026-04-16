// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type {
    BoolConfigurableParameter,
    ConfigurableParameter,
    ConfigurableParameterGroup,
    NumberConfigurableParameter,
    StringConfigurableParameter,
    TrainingConfigurationParameter,
} from '../src/constants/shared-types';

export function getMockedConfigurationParameter(
    parameter: Partial<NumberConfigurableParameter> & Required<Pick<NumberConfigurableParameter, 'value_type'>>
): NumberConfigurableParameter;
export function getMockedConfigurationParameter(
    parameter: Partial<StringConfigurableParameter> & Required<Pick<StringConfigurableParameter, 'value_type'>>
): StringConfigurableParameter;
export function getMockedConfigurationParameter(
    parameter: Partial<BoolConfigurableParameter> & Required<Pick<BoolConfigurableParameter, 'value_type'>>
): BoolConfigurableParameter;
export function getMockedConfigurationParameter(
    parameter: Partial<ConfigurableParameter> & Required<Pick<ConfigurableParameter, 'value_type'>> = {
        value_type: 'float',
    }
): ConfigurableParameter {
    if (parameter.value_type === 'float' || parameter.value_type === 'int') {
        return {
            type: 'parameter',
            value: 0,
            key: 'mocked_parameter',
            name: 'Mocked Parameter',
            max_value: 100,
            min_value: 0,
            description: 'This is a mocked configuration parameter',
            default_value: 50,
            allowed_values: null,
            depends_on: null,
            ...parameter,
        };
    }

    if (parameter.value_type === 'str') {
        return {
            type: 'parameter',
            value: 'mocked_value',
            key: 'mocked_string_parameter',
            name: 'Mocked String Parameter',
            description: 'This is a mocked string configuration parameter',
            default_value: 'default_mocked_value',
            allowed_values: null,
            depends_on: null,
            ...parameter,
        };
    }

    if (parameter.value_type === 'bool') {
        return {
            type: 'parameter',
            value: false,
            key: 'mocked_bool_parameter',
            name: 'Mocked Bool Parameter',
            description: 'This is a mocked boolean configuration parameter',
            default_value: false,
            depends_on: null,
            ...parameter,
        };
    }

    throw new Error(`Unsupported parameter type: ${parameter.value_type}`);
}

export const getMockedConfigurationParameterGroup = (
    overrides: Partial<ConfigurableParameterGroup> = {}
): ConfigurableParameterGroup => {
    return {
        type: 'parameter_group',
        key: 'mocked_group',
        name: 'Mocked Group',
        description: 'This is a mocked configuration parameter group',
        parameters: [getMockedConfigurationParameter({ value_type: 'float' })],
        ...overrides,
    };
};

export const getMockedTrainingConfiguration = (): TrainingConfigurationParameter[] => {
    const datasetPreparationGroup: ConfigurableParameterGroup = {
        type: 'parameter_group',
        key: 'dataset_preparation',
        name: 'Dataset preparation',
        description: '',
        parameters: [
            {
                type: 'parameter_group',
                key: 'augmentation',
                name: 'Data augmentation',
                description: '',
                parameters: [
                    {
                        type: 'parameter_group',
                        key: 'mosaic',
                        name: 'Mosaic',
                        description: '',
                        parameters: [
                            {
                                type: 'parameter',
                                key: 'enable',
                                name: 'Enable',
                                description: '',
                                value: true,
                                default_value: true,
                                value_type: 'bool',
                            },
                        ],
                    },
                    {
                        type: 'parameter_group',
                        key: 'gaussian_blur',
                        name: 'Gaussian blur',
                        description: '',
                        parameters: [
                            {
                                type: 'parameter',
                                key: 'sigma',
                                name: 'Sigma range',
                                description: '',
                                value: [0.1, 2],
                                default_value: [0.1, 2],
                                value_type: 'float_range',
                                min_value: 0,
                                max_value: 10,
                            },
                            {
                                type: 'parameter',
                                key: 'probability',
                                name: 'Probability',
                                description: '',
                                value: 0.5,
                                default_value: 0.5,
                                value_type: 'float',
                                min_value: 0,
                                max_value: 1,
                                allowed_values: null,
                            },
                        ],
                    },
                ],
            },
            {
                type: 'parameter_group',
                key: 'filtering',
                name: 'Filtering',
                description: '',
                parameters: [
                    {
                        type: 'parameter_group',
                        key: 'min_annotation_pixels',
                        name: 'Minimum annotation pixels',
                        description: '',
                        parameters: [
                            {
                                type: 'parameter',
                                key: 'value',
                                name: 'Minimum annotation pixels',
                                description: '',
                                value: 1,
                                default_value: 1,
                                value_type: 'int',
                                min_value: 0,
                                max_value: 200000000,
                                allowed_values: null,
                            },
                        ],
                    },
                ],
            },
        ],
    };

    const trainingGroup: ConfigurableParameterGroup = {
        type: 'parameter_group',
        key: 'training',
        name: 'Training',
        description: '',
        parameters: [
            {
                type: 'parameter',
                key: 'max_epochs',
                name: 'Maximum epochs',
                description: '',
                value: 200,
                default_value: 200,
                value_type: 'int',
                min_value: 0,
                max_value: null,
                allowed_values: null,
            },
        ],
    };

    return [datasetPreparationGroup, trainingGroup];
};
