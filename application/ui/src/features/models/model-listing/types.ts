// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Model } from '../../../constants/shared-types';

export type GroupByMode = 'dataset' | 'architecture';

export type SortBy = 'name' | 'trained' | 'architecture' | 'size' | 'score';

export type DatasetGroup = {
    id: string;
    name: string;
    createdAt: string;
    labelCount: number;
    imageCount: number;
    trainingSubsets: {
        training: number;
        validation: number;
        testing: number;
    };
};

export type ArchitectureGroup = {
    name: string;
    recommendedFor: string;
};

export type GroupedModels = {
    group: DatasetGroup | ArchitectureGroup;
    models: Model[];
};
