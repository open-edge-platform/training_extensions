// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { orderBy } from 'lodash-es';

import type { ModelArchitectureWithPerformanceCategory } from '../../../../constants/shared-types';

export const SortingOptions = {
    NAME_ASC: 'name-asc',
    NAME_DESC: 'name-desc',
    INFERENCE_SPEED_ASC: 'inference-speed-asc',
    INFERENCE_SPEED_DESC: 'inference-speed-desc',
    TRAINING_TIME_ASC: 'training-time-asc',
    TRAINING_TIME_DESC: 'training-time-desc',
    ACCURACY_ASC: 'accuracy-asc',
    ACCURACY_DESC: 'accuracy-desc',
} as const;

export type SortingOptions = (typeof SortingOptions)[keyof typeof SortingOptions];

type SortingHandler = (
    modelArchitectures: ModelArchitectureWithPerformanceCategory[]
) => ModelArchitectureWithPerformanceCategory[];

export const SORTING_HANDLERS: Record<SortingOptions, SortingHandler> = {
    [SortingOptions.ACCURACY_ASC]: (modelArchitectures) =>
        orderBy(modelArchitectures, (modelArchitecture) => modelArchitecture.stats.performance_ratings.accuracy, 'asc'),
    [SortingOptions.ACCURACY_DESC]: (modelArchitectures) =>
        orderBy(
            modelArchitectures,
            (modelArchitecture) => modelArchitecture.stats.performance_ratings.accuracy,
            'desc'
        ),
    [SortingOptions.INFERENCE_SPEED_ASC]: (modelArchitectures) =>
        orderBy(
            modelArchitectures,
            (modelArchitecture) => modelArchitecture.stats.performance_ratings.inference_speed,
            'asc'
        ),
    [SortingOptions.INFERENCE_SPEED_DESC]: (modelArchitectures) =>
        orderBy(
            modelArchitectures,
            (modelArchitecture) => modelArchitecture.stats.performance_ratings.inference_speed,
            'desc'
        ),
    [SortingOptions.TRAINING_TIME_ASC]: (modelArchitectures) =>
        orderBy(
            modelArchitectures,
            (modelArchitecture) => modelArchitecture.stats.performance_ratings.training_time,
            'asc'
        ),
    [SortingOptions.TRAINING_TIME_DESC]: (modelArchitectures) =>
        orderBy(
            modelArchitectures,
            (modelArchitecture) => modelArchitecture.stats.performance_ratings.training_time,
            'desc'
        ),
    [SortingOptions.NAME_ASC]: (modelArchitectures) =>
        orderBy(modelArchitectures, (modelArchitecture) => modelArchitecture.name, 'asc'),
    [SortingOptions.NAME_DESC]: (modelArchitectures) =>
        orderBy(modelArchitectures, (modelArchitecture) => modelArchitecture.name, 'desc'),
};

export const SORT_OPTIONS = [
    [
        {
            key: SortingOptions.NAME_ASC,
            name: 'Name',
        },
        {
            key: SortingOptions.NAME_DESC,
            name: 'Name',
        },
    ],
    [
        {
            key: SortingOptions.INFERENCE_SPEED_ASC,
            name: 'Inference speed',
        },
        {
            key: SortingOptions.INFERENCE_SPEED_DESC,
            name: 'Inference speed',
        },
    ],
    [
        {
            key: SortingOptions.TRAINING_TIME_ASC,
            name: 'Training time',
        },
        {
            key: SortingOptions.TRAINING_TIME_DESC,
            name: 'Training time',
        },
    ],
    [
        {
            key: SortingOptions.ACCURACY_ASC,
            name: 'Accuracy',
        },
        {
            key: SortingOptions.ACCURACY_DESC,
            name: 'Accuracy',
        },
    ],
];
