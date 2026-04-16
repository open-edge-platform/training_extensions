// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import type { DatasetRevision, Model } from '../../../../constants/shared-types';
import { useGetActiveModel } from '../../hooks/api/use-get-active-model.hook';
import { GroupByMode, GroupedModels, SortBy } from '../types';
import {
    filterBySearch,
    filterOutFailedModels,
    filterOutTrainingModels,
    groupModels,
    pinModel,
    removeEmpty,
    sortGroupedModels,
} from '../utils/model-transforms';

type UseGroupedModelsOptions = {
    groupBy: GroupByMode;
    sortBy: SortBy;
    pinActive: boolean;
    searchBy: string;
    datasetRevisions: DatasetRevision[];
    showFailedModels: boolean;
};

// Responsible for:
// - Filtering models based on searchBy query and failed status (only models that are not currently training)
// - Grouping models based on the selected grouping mode
// - Sorting models within each group based on the selected sorting criteria
// - Pinning the active model to the top of its group if the pinActive option is enabled
export const useGroupedModels = (models: Model[] | undefined, options: UseGroupedModelsOptions): GroupedModels[] => {
    const { groupBy, sortBy, pinActive, searchBy, datasetRevisions, showFailedModels } = options;
    const activeModel = useGetActiveModel();

    return useMemo(() => {
        if (!models) return [];

        const filteredByTraining = filterOutTrainingModels(models);
        const filteredByFailedModels = showFailedModels
            ? filteredByTraining
            : filterOutFailedModels(filteredByTraining);
        const filteredBySearch = filterBySearch(filteredByFailedModels, searchBy);
        const grouped = groupModels(filteredBySearch, groupBy, datasetRevisions);
        const sorted = sortGroupedModels(grouped, sortBy, datasetRevisions);
        const pinned = pinModel(sorted, pinActive ? activeModel?.id : undefined);

        return removeEmpty(pinned);
    }, [models, groupBy, sortBy, pinActive, activeModel?.id, searchBy, datasetRevisions, showFailedModels]);
};
