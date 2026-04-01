// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { dimensionValue, Grid, useViewMode, View } from '@geti/ui';
import { useGetDatasetItemsById } from 'hooks/use-get-dataset-items-by-id.hook';
import { useGetDatasetMediaItems } from 'hooks/use-get-dataset-media-items.hook';

import { DatasetItemAnnotationStatus } from '../../constants/shared-types';
import { Gallery } from '../../features/dataset/gallery/gallery.component';
import type { FilterByStatusKey } from '../../features/dataset/gallery/toolbar/filter-by-status/filter-by-status.component';
import { Toolbar } from '../../features/dataset/gallery/toolbar/toolbar.component';
import { ExportJobsList } from '../../features/dataset/import-export/export-jobs-list/export-jobs-list.component';
import { ImportJobsList } from '../../features/dataset/import-export/import-jobs-list/import-jobs-list.component';

export const Dataset = () => {
    const [viewMode, setViewMode] = useViewMode('dataset-gallery-view-mode');
    const [filterStatus, setFilterStatus] = useState<DatasetItemAnnotationStatus | null>(null);
    const mediaItemsResponse = useGetDatasetMediaItems({
        annotationStatus: filterStatus ?? undefined,
    });

    const responseDatasetItems = useGetDatasetItemsById({
        annotationStatus: filterStatus ?? undefined,
    });

    const handleFilterByStatusChange = (status: FilterByStatusKey) => {
        setFilterStatus(status === 'all' ? null : status);
    };

    const handleNextPageFetch = () => {
        if (mediaItemsResponse.hasNextPage && !mediaItemsResponse.isFetchingNextPage) {
            mediaItemsResponse.fetchNextPage();
        }

        if (responseDatasetItems.hasNextPage && !responseDatasetItems.isFetchingNextPage) {
            responseDatasetItems.fetchNextPage();
        }
    };

    const handleUserReviewedChange = (mediaItemId: string) => {
        return responseDatasetItems.reviewStatus.get(mediaItemId) ?? false;
    };

    return (
        <Grid
            height='100%'
            gridArea='content'
            rows={['auto', 'auto', '1fr']}
            UNSAFE_style={{ padding: dimensionValue('size-350'), paddingBottom: 0 }}
        >
            <View gridRow='1'>
                <ExportJobsList predicate={({ datasetId }) => datasetId === null} />
                <ImportJobsList />
            </View>

            <View gridRow='2'>
                <Toolbar
                    items={mediaItemsResponse.items}
                    viewMode={viewMode}
                    setViewMode={setViewMode}
                    onFilter={handleFilterByStatusChange}
                />
            </View>

            <View gridRow='3'>
                <Gallery
                    items={mediaItemsResponse.items}
                    viewMode={viewMode}
                    isPending={mediaItemsResponse.isPending || responseDatasetItems.isPending}
                    fetchNextPage={handleNextPageFetch}
                    isUserReviewed={handleUserReviewedChange}
                    hasActiveFilter={filterStatus !== null}
                    isFetchingNextPage={mediaItemsResponse.isFetchingNextPage}
                />
            </View>
        </Grid>
    );
};
