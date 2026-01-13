// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { useGetModels } from '../hooks/api/use-get-models.hook';
import { useGroupedModels } from './hooks/use-grouped-models.hook';
import { ModelListing } from './model-listing.component';
import { GroupByMode, SortBy } from './types';

export const ModelListingContainer = () => {
    const { data: models } = useGetModels();

    const [groupBy, setGroupBy] = useState<GroupByMode>('dataset');
    const [sortBy, setSortBy] = useState<SortBy>('score');
    const [pinActive, setPinActive] = useState<boolean>(false);

    const groupedModels = useGroupedModels(models, { groupBy, sortBy, pinActive });

    return (
        <ModelListing
            groupedModels={groupedModels}
            groupBy={groupBy}
            sortBy={sortBy}
            onGroupByChange={setGroupBy}
            onSortChange={(key) => setSortBy(key as SortBy)}
            onPinActiveToggle={() => setPinActive((prev) => !prev)}
        />
    );
};
