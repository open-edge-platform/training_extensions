// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Size } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { useSelectedData } from 'src/routes/dataset/provider';

import { MediaItem } from '../gallery/media-item.component';
import { MediaThumbnail } from '../gallery/media-thumbnail.component';
import { useGetDatasetItems } from '../gallery/use-get-dataset-items.hook';
import { getThumbnailUrl } from '../gallery/utils';
import { VirtualizerGridLayout } from '../virtualizer-grid-layout.component';

const layoutOptions = {
    maxColumns: 1,
    minSpace: new Size(8, 8),
    minItemSize: new Size(180, 180),
    preserveAspectRatio: true,
};

export const SidebarItems = () => {
    const project_id = useProjectIdentifier();
    const { mediaState, selectedKeys, setSelectedKeys } = useSelectedData();
    const { items, hasNextPage, isFetchingNextPage, fetchNextPage } = useGetDatasetItems();

    return (
        <VirtualizerGridLayout
            items={items}
            ariaLabel='sidebar-items'
            selectionMode='single'
            mediaState={mediaState}
            selectedKeys={selectedKeys}
            layoutOptions={layoutOptions}
            isLoadingMore={isFetchingNextPage}
            onLoadMore={() => hasNextPage && fetchNextPage()}
            onSelectionChange={setSelectedKeys}
            contentItem={(item) => (
                <MediaItem
                    contentElement={() => (
                        <MediaThumbnail alt={item.name} url={getThumbnailUrl(project_id, String(item.id))} />
                    )}
                />
            )}
        />
    );
};
