// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    BoolParameter,
    ConfigurationParameter,
    EnumConfigurationParameter,
    NumberParameter,
    TrainingConfiguration,
} from '../src/features/models/configuration.interface';

export function getMockedConfigurationParameter(
    parameter: Partial<EnumConfigurationParameter> & Required<Pick<EnumConfigurationParameter, 'type'>>
): EnumConfigurationParameter;
export function getMockedConfigurationParameter(
    parameter: Partial<NumberParameter> & Required<Pick<NumberParameter, 'type'>>
): NumberParameter;
export function getMockedConfigurationParameter(
    parameter: Partial<BoolParameter> & Required<Pick<BoolParameter, 'type'>>
): BoolParameter;
export function getMockedConfigurationParameter(
    parameter: Partial<ConfigurationParameter> & Required<Pick<ConfigurationParameter, 'type'>> = {
        type: 'float',
    }
): ConfigurationParameter {
    if (parameter.type === 'float' || parameter.type === 'int') {
        return {
            value: 0,
            key: 'mocked_parameter',
            name: 'Mocked Parameter',
            max_value: 100,
            min_value: 0,
            description: 'This is a mocked configuration parameter',
            default_value: 50,
            ...parameter,
        };
    }

    if (parameter.type === 'bool') {
        return {
            value: false,
            key: 'mocked_bool_parameter',
            name: 'Mocked Bool Parameter',
            description: 'This is a mocked boolean configuration parameter',
            default_value: false,
            ...parameter,
        };
    }

    if (parameter.type === 'enum') {
        return {
            allowed_values: [100, 200],
            default_value: 100,
            name: 'Mocked Enum Parameter',
            description: 'This is a mocked enum configuration parameter',
            value: 100,
            key: 'mocked_enum_parameter',
            ...parameter,
        };
    }

    throw new Error(`Unsupported parameter type: ${parameter.type}`);
}

export const getMockedTrainingConfiguration = (config: Partial<TrainingConfiguration> = {}): TrainingConfiguration => ({
    training: [
        getMockedConfigurationParameter({
            key: 'max_epochs',
            type: 'int',
            name: 'Maximum epochs',
            value: 200,
            description: 'Maximum number of training epochs to run',
            default_value: 500,
            max_value: null,
            min_value: 0,
        }),
        getMockedConfigurationParameter({
            key: 'learning_rate',
            type: 'float',
            name: 'Learning rate',
            value: 0.004,
            description: 'Base learning rate for the optimizer',
            default_value: 0.001,
            max_value: 1,
            min_value: 0,
        }),
        {
            early_stopping: [
                getMockedConfigurationParameter({
                    key: 'enable',
                    type: 'bool',
                    name: 'Enable early stopping',
                    value: true,
                    description: 'Whether to stop training early when performance stops improving',
                    default_value: true,
                }),
                getMockedConfigurationParameter({
                    key: 'patience',
                    type: 'int',
                    name: 'Patience',
                    value: 10,
                    description: 'Number of epochs with no improvement after which training will be stopped',
                    default_value: 1,
                    max_value: null,
                    min_value: 0,
                }),
            ],
        },
    ],
    dataset_preparation: {
        subset_split: [
            getMockedConfigurationParameter({
                key: 'training',
                type: 'int',
                name: 'Training percentage',
                value: 70,
                description: 'Percentage of data to use for training',
                default_value: 70,
                max_value: 100,
                min_value: 1,
            }),
            getMockedConfigurationParameter({
                key: 'validation',
                type: 'int',
                name: 'Validation percentage',
                value: 20,
                description: 'Percentage of data to use for validation',
                default_value: 20,
                max_value: 100,
                min_value: 1,
            }),
            getMockedConfigurationParameter({
                key: 'test',
                type: 'int',
                name: 'Test percentage',
                value: 10,
                description: 'Percentage of data to use for testing',
                default_value: 10,
                max_value: 100,
                min_value: 1,
            }),
            getMockedConfigurationParameter({
                key: 'dataset_size',
                type: 'int',
                name: 'Dataset size',
                value: 100,
                description: 'Dataset size',
                default_value: 100,
                max_value: null,
                min_value: 1,
            }),
        ],
        augmentation: {},
        filtering: {},
    },
    evaluation: [],
    ...config,
});
