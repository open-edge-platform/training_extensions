// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { partition } from 'lodash-es';

import type { DatasetRevision, Model } from '../../../../constants/shared-types';
import type { GroupByMode, GroupedModels, SortBy } from '../types';
import { groupModelsByArchitecture, groupModelsByDataset } from './grouping';
import { sortModels } from './sorting';
import { isFailedModel, isTrainingModel } from './utils';

export const filterBySearch = (models: Model[], query: string): Model[] =>
    query ? models.filter((model) => model.name?.toLowerCase().includes(query.toLowerCase())) : models;

export const filterOutFailedModels = (models: Model[]): Model[] => {
    return models.filter((model) => !isFailedModel(model));
};

export const filterOutTrainingModels = (models: Model[]): Model[] => {
    return models.filter((model) => !isTrainingModel(model));
};

export const groupModels = (
    models: Model[],
    mode: GroupByMode,
    datasetRevisions: DatasetRevision[]
): GroupedModels[] =>
    mode === 'dataset' ? groupModelsByDataset(models, { datasetRevisions }) : groupModelsByArchitecture(models);

export const sortGroupedModels = (
    groups: GroupedModels[],
    sortBy: SortBy,
    datasetRevisions: DatasetRevision[]
): GroupedModels[] => groups.map((group) => ({ ...group, models: sortModels(group.models, sortBy, datasetRevisions) }));

export const pinModel = (groups: GroupedModels[], modelId: string | undefined): GroupedModels[] => {
    if (!modelId) return groups;

    return groups.map((group) => {
        const [pinnedModel, otherModels] = partition(group.models, (model) => model.id === modelId);

        return { ...group, models: [...pinnedModel, ...otherModels] };
    });
};

export const removeEmpty = (groups: GroupedModels[]): GroupedModels[] =>
    groups.filter((group) => group.models.length > 0);
