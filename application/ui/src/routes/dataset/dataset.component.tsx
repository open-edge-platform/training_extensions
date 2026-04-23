// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { dimensionValue, Grid, useViewMode, View } from '@geti/ui';
import { useDatasetMediaWithReviewStatus } from 'hooks/use-dataset-media-with-review-status.hook';
import { useLabelsSearchParams } from 'hooks/use-labels-search-params.hook';

import { DatasetItemAnnotationStatus } from '../../constants/shared-types';
import { Gallery } from '../../features/dataset/gallery/gallery.component';
import type { FilterByStatusKey } from '../../features/dataset/gallery/toolbar/filter-by-status/filter-by-status.component';
import { Toolbar } from '../../features/dataset/gallery/toolbar/toolbar.component';
import { ExportJobsList } from '../../features/dataset/import-export/export-jobs-list/export-jobs-list.component';
import { ImportJobsList } from '../../features/dataset/import-export/import-jobs-list/import-jobs-list.component';

export const Dataset = () => {
    const [viewMode, setViewMode] = useViewMode('dataset-gallery-view-mode');
    const { selectedLabelIds: selectedFilterLabels } = useLabelsSearchParams();
    const [filterStatus, setFilterStatus] = useState<DatasetItemAnnotationStatus | null>(null);
    const { items, isPending, isFetchingNextPage, fetchNextPage, isMediaItemReviewedById } =
        useDatasetMediaWithReviewStatus({
            annotationStatus: filterStatus ?? undefined,
        });

    const hasActiveFilter = filterStatus !== null || selectedFilterLabels.length > 0;

    const handleFilterByStatusChange = (status: FilterByStatusKey) => {
        setFilterStatus(status === 'all' ? null : status);
    };

    return (
        <Grid
            height='100%'
            gridArea='content'
            rows={['auto', 'auto', 'minmax(0, 1fr)']}
            UNSAFE_style={{ padding: dimensionValue('size-350') }}
        >
            <View gridRow='1 / 2'>
                <ExportJobsList predicate={({ datasetId }) => datasetId === null} />
                <ImportJobsList />
            </View>

            <View gridRow='2 / 3'>
                <Toolbar
                    items={items}
                    viewMode={viewMode}
                    setViewMode={setViewMode}
                    onFilter={handleFilterByStatusChange}
                />
            </View>

            <View gridRow='3 / 4'>
                <Gallery
                    items={items}
                    viewMode={viewMode}
                    isPending={isPending}
                    fetchNextPage={fetchNextPage}
                    isMediaItemReviewedById={isMediaItemReviewedById}
                    hasActiveFilter={hasActiveFilter}
                    isFetchingNextPage={isFetchingNextPage}
                />
            </View>
        </Grid>
    );
};
