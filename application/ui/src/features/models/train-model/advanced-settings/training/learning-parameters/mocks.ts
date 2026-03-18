// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    getMockedConfigurationParameter,
    getMockedConfigurationParameterGroup,
} from 'mocks/mock-training-configuration';

import { LearningConfigurationGroup } from './utils';

export const learningParameters: LearningConfigurationGroup = getMockedConfigurationParameterGroup({
    key: 'training',
    name: 'Training',
    description: 'Configurable parameters related to the learning phase (hyperparameters).',
    parameters: [
        getMockedConfigurationParameter({
            value_type: 'int',
            key: 'max_epochs',
            name: 'Maximum epochs',
            description:
                'Maximum number of epochs to train the model. An epoch is one complete pass through the training ' +
                'dataset.',
            depends_on: null,
            value: 200,
            default_value: 200,
            min_value: 1,
            max_value: null,
            allowed_values: null,
        }),
        getMockedConfigurationParameter({
            value_type: 'int',
            key: 'batch_size',
            name: 'Batch size',
            description:
                "Number of training samples processed before the model's internal parameters are updated. A larger " +
                'batch size can speed up training but may require more memory, while a smaller batch size can help ' +
                'avoid OOM (Out of Memory) errors at the cost of longer training times and potentially noisier ' +
                'gradient estimates.',
            depends_on: null,
            value: 4,
            default_value: 8,
            min_value: 1,
            max_value: null,
            allowed_values: null,
        }),
        getMockedConfigurationParameterGroup({
            key: 'early_stopping',
            name: 'Early stopping',
            description:
                'Early stopping is a technique to prevent overfitting by stopping training when performance on a ' +
                'validation set stops improving.',
            parameters: [
                getMockedConfigurationParameter({
                    value_type: 'bool',
                    key: 'enable',
                    name: 'Enable',
                    description: 'Toggle to enable or disable early stopping during training.',
                    depends_on: null,
                    value: true,
                    default_value: true,
                }),
                getMockedConfigurationParameter({
                    value_type: 'int',
                    key: 'patience',
                    name: 'Patience',
                    description: 'Number of epochs with no improvement after which training will be stopped.',
                    depends_on: null,
                    value: 10,
                    default_value: 10,
                    min_value: 1,
                    max_value: null,
                    allowed_values: null,
                }),
            ],
        }),
        getMockedConfigurationParameter({
            value_type: 'float',
            key: 'learning_rate',
            name: 'Learning rate',
            description:
                'Learning rate for the optimizer, controlling the step size during model weight updates. A smaller ' +
                'learning rate may lead to more stable convergence, while a larger learning rate may speed up ' +
                'training but risk overshooting minima in the loss landscape.',
            depends_on: null,
            value: 0.004,
            default_value: 0.004,
            min_value: 0,
            max_value: 1,
            allowed_values: null,
        }),
        getMockedConfigurationParameter({
            value_type: 'float',
            key: 'weight_decay',
            name: 'Weight decay',
            description:
                'Weight decay is a regularization technique that adds a penalty to the loss function based on the ' +
                'squared magnitude of the model weights (L2 regularization). It helps prevent overfitting by ' +
                'discouraging large weight values.',
            depends_on: null,
            value: 0.0001,
            default_value: 0.0001,
            min_value: 0,
            max_value: 1,
            allowed_values: null,
        }),
        getMockedConfigurationParameterGroup({
            key: 'scheduler',
            name: 'Learning rate scheduler',
            description:
                'The learning rate scheduler adjusts the learning rate during training according to a predefined ' +
                'schedule or based on validation performance, helping to improve convergence and training stability.',
            parameters: [
                getMockedConfigurationParameter({
                    value_type: 'str',
                    key: 'type',
                    name: 'Scheduler type',
                    description:
                        'Type of learning rate scheduler to use during training. With ReduceLROnPlateau, the ' +
                        'learning rate will be reduced by a predetermined factor when the validation metric stops ' +
                        'improving. With CosineAnnealing, the learning rate will follow a cosine decay schedule, ' +
                        'gradually decreasing over the course of training.',
                    depends_on: null,
                    value: 'reduce_lr_on_plateau',
                    default_value: 'reduce_lr_on_plateau',
                    allowed_values: ['reduce_lr_on_plateau', 'cosine_annealing'],
                }),
                getMockedConfigurationParameterGroup({
                    key: 'warmup',
                    name: 'Learning rate linear warmup',
                    description:
                        'Learning rate warmup is a technique where the learning rate starts at a lower value and ' +
                        'gradually increases to the initial learning rate over a specified number of epochs at the ' +
                        'beginning of training. This can help stabilize training and improve convergence, especially ' +
                        'when using large learning rates or training on complex datasets.',
                    parameters: [
                        getMockedConfigurationParameter({
                            value_type: 'bool',
                            key: 'enable',
                            name: 'Enable',
                            description:
                                'Toggle to enable or disable the LR linear warmup phase at the beginning of training.',
                            depends_on: null,
                            value: false,
                            default_value: false,
                        }),
                        getMockedConfigurationParameter({
                            value_type: 'int',
                            key: 'epochs',
                            name: 'Warmup epochs',
                            description: 'Number of epochs for the LR linear warmup phase.',
                            depends_on: null,
                            value: 5,
                            default_value: 5,
                            min_value: 1,
                            max_value: null,
                            allowed_values: null,
                        }),
                    ],
                }),
                getMockedConfigurationParameter({
                    value_type: 'float',
                    key: 'factor',
                    name: 'Factor',
                    description: 'Factor by which the learning rate will be reduced. new_lr = lr * factor.',
                    depends_on: {
                        type: 'reduce_lr_on_plateau',
                    },
                    value: 0.5,
                    default_value: 0.1,
                    min_value: 0,
                    max_value: 1,
                    allowed_values: null,
                }),
                getMockedConfigurationParameter({
                    value_type: 'int',
                    key: 'patience',
                    name: 'Patience',
                    description: 'Number of epochs with no improvement after which learning rate will be reduced.',
                    depends_on: {
                        type: 'reduce_lr_on_plateau',
                    },
                    value: 5,
                    default_value: 4,
                    min_value: 1,
                    max_value: null,
                    allowed_values: null,
                }),
                getMockedConfigurationParameter({
                    value_type: 'float',
                    key: 'min_lr',
                    name: 'Minimum learning rate',
                    description: 'Minimum learning rate after annealing.',
                    depends_on: {
                        type: 'cosine_annealing',
                    },
                    value: 0,
                    default_value: 0,
                    min_value: 0,
                    max_value: 1,
                    allowed_values: null,
                }),
            ],
        }),
        getMockedConfigurationParameterGroup({
            key: 'gradient_accumulation',
            name: 'Gradient accumulation',
            description:
                'Gradient accumulation allows simulating larger batch sizes by accumulating gradients over multiple ' +
                'forward/backward passes before updating the model weights.',
            parameters: [
                getMockedConfigurationParameter({
                    value_type: 'bool',
                    key: 'enable',
                    name: 'Enable',
                    description: 'Toggle to enable or disable gradient accumulation during training.',
                    depends_on: null,
                    value: false,
                    default_value: false,
                }),
                getMockedConfigurationParameter({
                    value_type: 'int',
                    key: 'batches',
                    name: 'Gradient accumulation batches',
                    description:
                        'Number of steps (batches) to accumulate gradients before performing gradient descent step. ' +
                        'Effective batch size during training: batch_size * accumulate_grad_batches.',
                    depends_on: null,
                    value: 1,
                    default_value: 1,
                    min_value: 1,
                    max_value: null,
                    allowed_values: null,
                }),
            ],
        }),
        getMockedConfigurationParameterGroup({
            key: 'gradient_clip',
            name: 'Gradient clipping',
            description:
                'Gradient clipping prevents exploding gradients by capping gradient norms during backpropagation.',
            parameters: [
                getMockedConfigurationParameter({
                    value_type: 'bool',
                    key: 'enable',
                    name: 'Enable',
                    description: 'Toggle to enable or disable gradient clipping during training.',
                    depends_on: null,
                    value: false,
                    default_value: true,
                }),
                getMockedConfigurationParameter({
                    value_type: 'float',
                    key: 'max_grad_norm',
                    name: 'Maximum gradient L2 norm',
                    description:
                        'Maximum L2 norm of the gradients. Gradients with norm larger than this value will be clipped.',
                    depends_on: null,
                    value: 1,
                    default_value: 35,
                    min_value: 0,
                    max_value: null,
                    allowed_values: null,
                }),
            ],
        }),
        getMockedConfigurationParameter({
            type: 'parameter',
            key: 'input_size_width',
            name: 'Input size width',
            description:
                'Width size in pixels for model input images. Determines the horizontal resolution at which images ' +
                'are processed.',
            depends_on: null,
            value_type: 'int',
            value: 992,
            default_value: 992,
            min_value: 0,
            max_value: null,
            allowed_values: [128, 256, 384, 512, 640, 800, 992, 1024],
        }),
        getMockedConfigurationParameter({
            type: 'parameter',
            key: 'input_size_height',
            name: 'Input size height',
            description:
                'Height size in pixels for model input images. Determines the vertical resolution at which images ' +
                'are processed.',
            depends_on: null,
            value_type: 'int',
            value: 800,
            default_value: 800,
            min_value: 0,
            max_value: null,
            allowed_values: [128, 256, 384, 512, 640, 800, 992, 1024],
        }),
    ],
}) as LearningConfigurationGroup;
