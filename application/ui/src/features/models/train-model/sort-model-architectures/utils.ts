// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { orderBy } from 'lodash-es';

import type { ModelArchitectureWithPerformanceCategory } from '../../../../constants/shared-types';

export const SortingOptions = {
    NAME_ASC: 'name-asc',
    NAME_DESC: 'name-desc',
    SIZE_ASC: 'size-asc',
    SIZE_DESC: 'size-desc',
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
    [SortingOptions.NAME_ASC]: (modelArchitectures) =>
        orderBy(modelArchitectures, (modelArchitecture) => modelArchitecture.name, 'asc'),
    [SortingOptions.NAME_DESC]: (modelArchitectures) =>
        orderBy(modelArchitectures, (modelArchitecture) => modelArchitecture.name, 'desc'),
    [SortingOptions.SIZE_ASC]: (modelArchitectures) =>
        orderBy(modelArchitectures, (modelArchitecture) => modelArchitecture.stats.trainable_parameters, 'asc'),
    [SortingOptions.SIZE_DESC]: (modelArchitectures) =>
        orderBy(modelArchitectures, (modelArchitecture) => modelArchitecture.stats.trainable_parameters, 'desc'),
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
            key: SortingOptions.SIZE_ASC,
            name: 'Size',
        },
        {
            key: SortingOptions.SIZE_DESC,
            name: 'Size',
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
