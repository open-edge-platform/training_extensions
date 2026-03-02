// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef } from 'react';

import { Size, useUnwrapDOMRef, View } from '@geti/ui';

import { VirtualizerGridLayout } from '../../../../components/virtualizer-grid-layout/virtualizer-grid-layout.component';
import type { Media } from '../../../../constants/shared-types';
import { useGetDatasetItemsById } from '../../../../hooks/use-get-dataset-items-by-id.hook';
import { SIDEBAR_MEDIA_SIZE } from '../constants';
import { SidebarMediaItem } from './sidebar-media-item.component';
import { useKeyboardNavigation } from './use-keyboard-navigation.hook';

const layoutOptions = {
    maxColumns: 1,
    minSpace: new Size(8, 8),
    minItemSize: new Size(SIDEBAR_MEDIA_SIZE, SIDEBAR_MEDIA_SIZE),
    maxItemSize: new Size(SIDEBAR_MEDIA_SIZE, SIDEBAR_MEDIA_SIZE),
    preserveAspectRatio: true,
};

type SidebarItemsProps = {
    items: Media[];
    hasNextPage: boolean;
    isFetchingNextPage: boolean;
    mediaItem: Media;
    fetchNextPage: () => void;
    onSelectedMediaItem: (item: Media) => void;
};

export const SidebarItems = ({
    mediaItem,
    items,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
    onSelectedMediaItem,
}: SidebarItemsProps) => {
    const ref = useRef(null);
    const unwrapRef = useUnwrapDOMRef(ref);
    const { datasetItemsById } = useGetDatasetItemsById({ limit: Math.max(items.length, 1) });

    const selectedIndex = items.findIndex((item) => item.id === mediaItem.id);

    useKeyboardNavigation({
        ref: unwrapRef,
        items,
        selectedIndex,
        onSelectedMediaItem,
    });

    return (
        <View ref={ref} width={'100%'} height={'100%'}>
            <VirtualizerGridLayout
                items={items}
                ariaLabel='sidebar-items'
                selectionMode='single'
                selectedKeys={new Set([String(mediaItem.id)])}
                layoutOptions={layoutOptions}
                isLoadingMore={isFetchingNextPage}
                scrollToIndex={selectedIndex}
                onLoadMore={() => hasNextPage && fetchNextPage()}
                contentItem={(item) => {
                    const itemId = String(item.id);
                    const isUserReviewed = datasetItemsById.get(itemId) ?? false;

                    return (
                        <SidebarMediaItem
                            item={item}
                            onSelectedMediaItem={onSelectedMediaItem}
                            isUserReviewed={isUserReviewed}
                        />
                    );
                }}
            />
        </View>
    );
};
