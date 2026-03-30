// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { dimensionValue, Grid, useViewMode, View } from '@geti/ui';
import { useGetDatasetMediaItems } from 'hooks/use-get-dataset-media-items.hook';

import { DatasetItemAnnotationStatus } from '../../constants/shared-types';
import { useMediaUpload } from '../../features/dataset/api/use-media-upload';
import { Gallery } from '../../features/dataset/gallery/gallery.component';
import type { FilterByStatusKey } from '../../features/dataset/gallery/toolbar/filter-by-status/filter-by-status.component';
import { Toolbar } from '../../features/dataset/gallery/toolbar/toolbar.component';
import { ExportJobsList } from '../../features/dataset/import-export/export-jobs-list/export-jobs-list.component';
import { ImportJobsList } from '../../features/dataset/import-export/import-jobs-list/import-jobs-list.component';

export const Dataset = () => {
    const [viewMode, setViewMode] = useViewMode('dataset-gallery-view-mode');
    const [filterStatus, setFilterStatus] = useState<DatasetItemAnnotationStatus | null>(null);
    const { items, hasNextPage, isFetchingNextPage, fetchNextPage, isPending } = useGetDatasetMediaItems({
        annotationStatus: filterStatus ?? undefined,
    });
    const { uploadMedia } = useMediaUpload();

    const handleFilterByStatusChange = (status: FilterByStatusKey) => {
        if (status === 'all') {
            setFilterStatus(null);

            return;
        }

        setFilterStatus(status);
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
                    items={items}
                    viewMode={viewMode}
                    setViewMode={setViewMode}
                    onFilter={handleFilterByStatusChange}
                />
            </View>

            <View gridRow='3'>
                <Gallery
                    items={items}
                    annotationStatus={filterStatus ?? undefined}
                    viewMode={viewMode}
                    isPending={isPending}
                    hasActiveFilter={filterStatus !== null}
                    fetchNextPage={fetchNextPage}
                    hasNextPage={hasNextPage}
                    isFetchingNextPage={isFetchingNextPage}
                    onFilesDropped={uploadMedia}
                />
            </View>
        </Grid>
    );
};
