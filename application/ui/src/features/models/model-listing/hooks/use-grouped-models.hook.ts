// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { orderBy } from 'lodash-es';

import type { DatasetRevision, Model } from '../../../../constants/shared-types';
import { GroupByMode, GroupedModels, SortBy } from '../types';
import {
    filterBySearch,
    filterOutFailedModels,
    filterOutTrainingModels,
    groupModels,
    removeEmpty,
    sortGroupedModels,
} from '../utils/model-transforms';
import { sortModels } from '../utils/sorting';

type UseGroupedModelsOptions = {
    groupBy: GroupByMode;
    sortBy: SortBy;
    searchBy: string;
    datasetRevisions: DatasetRevision[];
    showFailedModels: boolean;
};

// Responsible for:
// - Filtering models based on searchBy query and failed status (only models that are not currently training)
// - Grouping models based on the selected grouping mode
// - Sorting models within each group based on the selected sorting criteria
export const useGroupedModels = (models: Model[] | undefined, options: UseGroupedModelsOptions): GroupedModels[] => {
    const { groupBy, sortBy, searchBy, datasetRevisions, showFailedModels } = options;

    return useMemo(() => {
        if (!models) return [];

        const filteredByTraining = filterOutTrainingModels(models);
        const filteredByFailedModels = showFailedModels
            ? filteredByTraining
            : filterOutFailedModels(filteredByTraining);
        const filteredBySearch = filterBySearch(filteredByFailedModels, searchBy);
        const grouped = groupModels(filteredBySearch, groupBy, datasetRevisions);
        const sortedModelsInsideGroup = sortGroupedModels(grouped, sortBy, datasetRevisions);
        const sortedGroupsByDatasetRevisionDate = orderBy(
            sortedModelsInsideGroup,
            (group) => {
                if (group.group.type === 'dataset') {
                    return group.group.createdAt;
                }

                const mostRecentModel = sortModels(group.models, 'dataset', datasetRevisions)?.at(0);
                const datasetRevisionDate = datasetRevisions.find(
                    (datasetRevision) => mostRecentModel?.training_info.dataset_revision_id === datasetRevision.id
                )?.created_at;

                return datasetRevisionDate;
            },
            'desc'
        );

        return removeEmpty(sortedGroupsByDatasetRevisionDate);
    }, [models, groupBy, sortBy, searchBy, datasetRevisions, showFailedModels]);
};
