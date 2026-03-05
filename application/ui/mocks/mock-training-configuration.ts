// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ConfigurableParameterGroup, TrainingConfigurationParameter } from '../src/constants/shared-types';

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
                                min_value: null,
                                max_value: null,
                                allowed_values: null,
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
                                min_value: null,
                                max_value: null,
                                allowed_values: null,
                            },
                            {
                                type: 'parameter',
                                key: 'probability',
                                name: 'Probability',
                                description: '',
                                value: null,
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
