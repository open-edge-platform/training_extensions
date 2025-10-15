// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Selection, Size } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { DatasetItem } from 'src/features/annotator/types';
import { useSelectedData } from 'src/routes/dataset/provider';

import { MediaItem } from '../gallery/media-item.component';
import { MediaThumbnail } from '../gallery/media-thumbnail.component';
import { useGetDatasetItems } from '../gallery/use-get-dataset-items.hook';
import { getThumbnailUrl } from '../gallery/utils';
import { VirtualizerGridLayout } from '../virtualizer-grid-layout/virtualizer-grid-layout.component';

const layoutOptions = {
    maxColumns: 1,
    minSpace: new Size(8, 8),
    minItemSize: new Size(180, 180),
    preserveAspectRatio: true,
};

type SidebarItemsProps = {
    mediaItem: DatasetItem;
    onSelectedMediaItem: (item: DatasetItem) => void;
};

export const SidebarItems = ({ mediaItem, onSelectedMediaItem }: SidebarItemsProps) => {
    const project_id = useProjectIdentifier();
    const { mediaState } = useSelectedData();
    const [selectedKeys, setSelectedKeys] = useState<Selection>(new Set([String(mediaItem.id)]));
    const { items, hasNextPage, isFetchingNextPage, fetchNextPage } = useGetDatasetItems();

    const selectedIndex = items.findIndex((item) => item.id === mediaItem.id);

    return (
        <VirtualizerGridLayout
            items={items}
            ariaLabel='sidebar-items'
            selectionMode='single'
            mediaState={mediaState}
            selectedKeys={selectedKeys}
            layoutOptions={layoutOptions}
            isLoadingMore={isFetchingNextPage}
            scrollToIndex={selectedIndex}
            onLoadMore={() => hasNextPage && fetchNextPage()}
            onSelectionChange={setSelectedKeys}
            contentItem={(item) => (
                <MediaItem
                    contentElement={() => (
                        <MediaThumbnail
                            alt={item.name}
                            url={getThumbnailUrl(project_id, String(item.id))}
                            onClick={() => onSelectedMediaItem(item)}
                        />
                    )}
                />
            )}
        />
    );
};
