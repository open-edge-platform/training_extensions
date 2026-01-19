// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useState } from 'react';

import { useGetActiveModelId } from '../../hooks/api/use-get-active-model-id.hook';
import { useGetModels } from '../../hooks/api/use-get-models.hook';
import { useGroupedModels } from '../hooks/use-grouped-models.hook';
import type { GroupByMode, GroupedModels, SortBy } from '../types';

interface ModelListingContextValue {
    // State
    groupBy: GroupByMode;
    sortBy: SortBy;
    pinActive: boolean;
    expandedModelIds: Set<string>;
    activeModelId: string | undefined;
    groupedModels: GroupedModels[];
    searchBy: string;

    // Actions
    onGroupByChange: (mode: GroupByMode) => void;
    onSortChange: (key: SortBy) => void;
    onPinActiveToggle: () => void;
    onExpandModel: (modelId: string) => void;
    onSearchChange: (query: string) => void;
}

const ModelListingContext = createContext<ModelListingContextValue | null>(null);

interface ModelListingProviderProps {
    children: ReactNode;
}

export const ModelListingProvider = ({ children }: ModelListingProviderProps) => {
    const [groupBy, setGroupBy] = useState<GroupByMode>('dataset');
    const [sortBy, setSortBy] = useState<SortBy>('score');
    const [pinActive, setPinActive] = useState<boolean>(false);
    const [expandedModelIds, setExpandedModelIds] = useState<Set<string>>(new Set());
    const [searchBy, setSearchBy] = useState<string>('');

    const activeModelId = useGetActiveModelId();
    const { data: models } = useGetModels();
    const groupedModels = useGroupedModels(models, { groupBy, sortBy, pinActive, searchBy });

    const onGroupByChange = (mode: GroupByMode) => {
        setGroupBy(mode);
    };

    const onSortChange = (key: SortBy) => {
        setSortBy(key);
    };

    const onPinActiveToggle = () => {
        setPinActive((prev) => !prev);
    };

    const onSearchChange = (query: string) => {
        setSearchBy(query);
    };

    const onExpandModel = (modelId: string) => {
        setExpandedModelIds((prev) => {
            const newExpandedModelIds = new Set(prev);

            if (newExpandedModelIds.has(modelId)) {
                newExpandedModelIds.delete(modelId);
            } else {
                newExpandedModelIds.add(modelId);
            }

            return newExpandedModelIds;
        });
    };

    const value: ModelListingContextValue = {
        groupBy,
        sortBy,
        pinActive,
        expandedModelIds,
        activeModelId,
        groupedModels,
        searchBy,
        onGroupByChange,
        onSortChange,
        onPinActiveToggle,
        onExpandModel,
        onSearchChange,
    };

    return <ModelListingContext.Provider value={value}>{children}</ModelListingContext.Provider>;
};

export const useModelListing = (): ModelListingContextValue => {
    const context = useContext(ModelListingContext);

    if (!context) {
        throw new Error('useModelListing must be used within a ModelListingProvider');
    }

    return context;
};
