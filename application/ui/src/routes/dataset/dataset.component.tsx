// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Grid, useViewMode, View } from '@geti/ui';
import { useGetDatasetMediaItems } from 'hooks/use-get-dataset-media-items.hook';

import { Gallery } from '../../features/dataset/gallery/gallery.component';
import { Toolbar } from '../../features/dataset/gallery/toolbar/toolbar.component';
import { ExportJobsList } from '../../features/dataset/import-export/export-jobs-list/export-jobs-list.component';

const isMainDataset = <T extends { datasetId: string | null }>({ datasetId }: T) => datasetId === null;

export const Dataset = () => {
    const [viewMode, setViewMode] = useViewMode('dataset-gallery-view-mode');
    const { items, hasNextPage, isFetchingNextPage, fetchNextPage, isPending } = useGetDatasetMediaItems();

    return (
        <Grid
            height='100%'
            gridArea='content'
            rows={['auto', 'auto', '1fr']}
            UNSAFE_style={{ padding: dimensionValue('size-350'), paddingBottom: 0 }}
        >
            <View gridRow='1'>
                <ExportJobsList predicate={isMainDataset} />
            </View>

            <View gridRow='2'>
                <Toolbar items={items} viewMode={viewMode} setViewMode={setViewMode} />
            </View>

            <View gridRow='3'>
                <Gallery
                    items={items}
                    viewMode={viewMode}
                    isPending={isPending}
                    fetchNextPage={fetchNextPage}
                    hasNextPage={hasNextPage}
                    isFetchingNextPage={isFetchingNextPage}
                />
            </View>
        </Grid>
    );
};
