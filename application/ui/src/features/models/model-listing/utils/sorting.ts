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
            return orderBy(models, (model) => model.name?.toLowerCase() ?? '', 'asc');
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
        case 'dataset':
            const datasetRevisionsMap = new Map(
                datasetRevisions.map((datasetRevision) => [datasetRevision.id, datasetRevision.name])
            );
            return orderBy(
                models,
                [
                    // First we sort by models that have dataset revision so they are on the top.
                    (model) => {
                        const name =
                            model.training_info?.dataset_revision_id != null
                                ? datasetRevisionsMap.get(model.training_info.dataset_revision_id)
                                : undefined;

                        return name != null ? 0 : 1;
                    },
                    // Then we sort alphabetically by dataset name
                    (model) => {
                        if (model.training_info?.dataset_revision_id != null) {
                            return (
                                datasetRevisionsMap.get(model.training_info.dataset_revision_id)?.toLowerCase() ?? ''
                            );
                        }

                        return '';
                    },
                ],
                ['asc', 'asc']
            );
        default:
            console.error(`Unknown sort option: ${sortBy satisfies never}`);
            return models;
    }
};
