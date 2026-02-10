// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Flex, useViewMode } from '@geti/ui';
import { useGetDatasetMediaItems } from 'hooks/use-get-dataset-media-items.hook';

import { Gallery } from '../../features/dataset/gallery/gallery.component';
import { Toolbar } from '../../features/dataset/gallery/toolbar/toolbar.component';
import { ExportJobsList } from '../../features/dataset/import-export/export-jobs-list/export-jobs-list.component';

export const Dataset = () => {
    const [viewMode, setViewMode] = useViewMode('dataset-gallery-view-mode');
    const { items, hasNextPage, isFetchingNextPage, fetchNextPage, isPending } = useGetDatasetMediaItems();

    return (
        <Flex
            height='100%'
            gridArea='content'
            direction='column'
            UNSAFE_style={{ padding: dimensionValue('size-350'), paddingBottom: 0 }}
        >
            <ExportJobsList />

            <Toolbar items={items} viewMode={viewMode} setViewMode={setViewMode} />

            <Gallery
                items={items}
                viewMode={viewMode}
                isPending={isPending}
                fetchNextPage={fetchNextPage}
                hasNextPage={hasNextPage}
                isFetchingNextPage={isFetchingNextPage}
            />
        </Flex>
    );
};
