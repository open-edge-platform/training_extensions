// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { SchemaModelView } from '../../../api/openapi-spec';

export type GroupByMode = 'dataset' | 'architecture';

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
    models: SchemaModelView[];
};
