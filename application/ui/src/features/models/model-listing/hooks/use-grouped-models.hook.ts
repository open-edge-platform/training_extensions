// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { partition } from 'lodash-es';

import { Model } from '../../../../constants/shared-types';
import { useGetActiveModelId } from '../../hooks/api/use-get-active-model-id.hook';
import { GroupByMode, GroupedModels, SortBy } from '../types';
import { groupModelsByArchitecture, groupModelsByDataset } from '../utils/grouping';
import { sortModels } from '../utils/sorting';

type UseGroupedModelsOptions = {
    groupBy: GroupByMode;
    sortBy: SortBy;
    pinActive: boolean;
};

export const useGroupedModels = (models: Model[] | undefined, options: UseGroupedModelsOptions): GroupedModels[] => {
    const { groupBy, sortBy, pinActive } = options;
    const activeModelId = useGetActiveModelId();

    return useMemo(() => {
        if (!models) return [];

        const groups = groupBy === 'dataset' ? groupModelsByDataset(models) : groupModelsByArchitecture(models);

        return groups.map((group) => {
            const sortedModels = sortModels(group.models, sortBy);

            if (!pinActive || !activeModelId) {
                return { ...group, models: sortedModels };
            }

            const [activeModel, otherModels] = partition(sortedModels, (model) => model.id === activeModelId);

            return { ...group, models: [...activeModel, ...otherModels] };
        });
    }, [models, groupBy, sortBy, pinActive, activeModelId]);
};
