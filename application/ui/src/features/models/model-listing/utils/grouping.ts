// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { SchemaModelView } from '../../../../api/openapi-spec';
import type { GroupedModels } from '../types';

export const groupModelsByDataset = (models: SchemaModelView[]): GroupedModels[] => {
    const groups: Record<string, GroupedModels> = {}; // datasetId -> models

    models.forEach((model) => {
        const datasetId = model.training_info.dataset_revision_id ?? 'unknown';
        const labels = model.training_info.label_schema_revision?.labels;
        const labelCount = Array.isArray(labels) ? labels.length : 0;

        if (!groups[datasetId]) {
            groups[datasetId] = {
                group: {
                    id: datasetId,
                    name: `Dataset #${datasetId}`,
                    createdAt: 'Created 01 Oct 2025, 11:07 AM',
                    labelCount,
                    imageCount: 3600,
                    trainingSubsets: {
                        training: 70,
                        validation: 20,
                        testing: 10,
                    },
                },
                models: [],
            };
        }

        groups[datasetId].models.push(model);
    });

    return Object.values(groups);
};

export const groupModelsByArchitecture = (models: SchemaModelView[]): GroupedModels[] => {
    const groups: Record<string, GroupedModels> = {}; // architecture -> models

    models.forEach((model) => {
        const arch = model.architecture;

        if (!groups[arch]) {
            groups[arch] = {
                group: {
                    name: arch,
                    recommendedFor: 'balance',
                },
                models: [],
            };
        }

        groups[arch].models.push(model);
    });

    return Object.values(groups);
};
