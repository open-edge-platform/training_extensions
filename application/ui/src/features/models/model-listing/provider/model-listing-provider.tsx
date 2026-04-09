// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useState } from 'react';

import { useGetDatasetRevisions } from 'hooks/use-get-dataset-revisions.hook';

import { type DatasetRevision } from '../../../../constants/shared-types';
import { useGetActiveModel } from '../../hooks/api/use-get-active-model.hook';
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
    datasetRevisions: DatasetRevision[];
    showFailedModels: boolean;

    // Actions
    onGroupByChange: (mode: GroupByMode) => void;
    onSortChange: (key: SortBy) => void;
    onPinActiveToggle: () => void;
    onExpandModel: (modelId: string) => void;
    onSearchChange: (query: string) => void;
    onToggleShowFailedModels: () => void;
}

const ModelListingContext = createContext<ModelListingContextValue | null>(null);

interface ModelListingProviderProps {
    children: ReactNode;
}

export const ModelListingProvider = ({ children }: ModelListingProviderProps) => {
    const [groupBy, setGroupBy] = useState<GroupByMode>('dataset');
    const [sortBy, setSortBy] = useState<SortBy>('score');
    const [pinActive, setPinActive] = useState<boolean>(false);
    const [showFailedModels, setShowFailedModels] = useState<boolean>(true);
    const [expandedModelIds, setExpandedModelIds] = useState<Set<string>>(new Set());
    const [searchBy, setSearchBy] = useState<string>('');

    const activeModel = useGetActiveModel();
    const { data: models } = useGetModels();
    const { data: datasetRevisions = [] } = useGetDatasetRevisions();
    const groupedModels = useGroupedModels(models, {
        groupBy,
        sortBy,
        pinActive,
        searchBy,
        datasetRevisions,
        showFailedModels,
    });

    const onGroupByChange = (mode: GroupByMode) => {
        if (mode === 'dataset' && sortBy === 'dataset') {
            setSortBy('architecture');
        } else if (mode === 'architecture' && sortBy === 'architecture') {
            setSortBy('dataset');
        }
        setGroupBy(mode);
    };

    const onSortChange = (key: SortBy) => {
        setSortBy(key);
    };

    const onPinActiveToggle = () => {
        setPinActive((prev) => !prev);
    };

    const toggleShowFailedModels = () => {
        setShowFailedModels((prev) => !prev);
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
        activeModelId: activeModel?.id,
        groupedModels,
        searchBy,
        datasetRevisions,
        showFailedModels,

        onGroupByChange,
        onSortChange,
        onPinActiveToggle,
        onExpandModel,
        onSearchChange,
        onToggleShowFailedModels: toggleShowFailedModels,
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
