// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import dayjs from 'dayjs';
import { orderBy } from 'lodash-es';

import type { DatasetRevision, Model } from '../../../../constants/shared-types';
import { getTestingMetric } from '../components/model-row/utils';
import type { SortBy } from '../types';

export const sortModels = (models: Model[], sortBy: SortBy, datasetRevisions: DatasetRevision[]): Model[] => {
    switch (sortBy) {
        case 'name':
            return orderBy(models, (model) => model.name.toLowerCase(), 'asc');
        case 'architecture':
            return orderBy(models, (model) => model.architecture?.toLowerCase() ?? '', 'asc');
        case 'trained':
            return orderBy(
                models,
                (model) => {
                    const date = dayjs(model.training_info?.end_time);

                    return date.isValid() ? date.valueOf() : 0;
                },
                'desc'
            );
        case 'size':
            return orderBy(models, (model) => model.size ?? 0, 'asc');
        case 'score':
            return orderBy(models, (model) => getTestingMetric(model)?.value ?? 0, 'desc');
        case 'dataset': {
            const datasetRevisionsMap = new Map(
                datasetRevisions.map((datasetRevision) => [datasetRevision.id, datasetRevision])
            );

            const getDatasetRevision = (model: Model) => {
                const id = model.training_info?.dataset_revision_id;
                return id != null ? datasetRevisionsMap.get(id) : undefined;
            };

            return orderBy(
                models,
                [
                    // First: models with a resolvable dataset revision come first.
                    (model) => (getDatasetRevision(model) != null ? 1 : 0),
                    // Second: sort by dataset revision creation date, newest first.
                    (model) => {
                        const createdAt = getDatasetRevision(model)?.created_at;

                        return createdAt ?? 0;
                    },
                    // Third: sort by dataset revision name, Z -> A.
                    (model) => getDatasetRevision(model)?.name?.toLowerCase() ?? '',
                ],
                ['desc', 'desc', 'desc']
            );
        }
        default:
            console.error(`Unknown sort option: ${sortBy satisfies never}`);
            return models;
    }
};
