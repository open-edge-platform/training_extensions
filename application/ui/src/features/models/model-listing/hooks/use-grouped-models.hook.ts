// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { Model } from '../../../../constants/shared-types';
import { useGetActiveModelArchitectureId } from '../../hooks/api/use-get-active-model-architecture-id.hook';
import { GroupByMode, GroupedModels, SortBy } from '../types';
import { filterBySearch, groupModels, pinModel, removeEmpty, sortGroupedModels } from '../utils/model-transforms';

type UseGroupedModelsOptions = {
    groupBy: GroupByMode;
    sortBy: SortBy;
    pinActive: boolean;
    searchBy: string;
};

// Responsible for:
// - Filtering models based on searchBy query
// - Grouping models based on the selected grouping mode
// - Sorting models within each group based on the selected sorting criteria
// - Pinning the active model to the top of its group if the pinActive option is enabled
export const useGroupedModels = (models: Model[] | undefined, options: UseGroupedModelsOptions): GroupedModels[] => {
    const { groupBy, sortBy, pinActive, searchBy } = options;
    const activeModelArchitectureId = useGetActiveModelArchitectureId();

    return useMemo(() => {
        if (!models) return [];

        const filtered = filterBySearch(models, searchBy);
        const grouped = groupModels(filtered, groupBy);
        const sorted = sortGroupedModels(grouped, sortBy);
        const pinned = pinModel(sorted, pinActive ? activeModelArchitectureId : undefined);

        return removeEmpty(pinned);
    }, [models, groupBy, sortBy, pinActive, activeModelArchitectureId, searchBy]);
};
