// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef } from 'react';

import { Size, useUnwrapDOMRef, View } from '@geti/ui';
import type { DatasetItem } from 'src/constants/shared-types';

import { useSelectedData } from '../../selected-data-provider.component';
import { VirtualizerGridLayout } from '../../virtualizer-grid-layout/virtualizer-grid-layout.component';
import { SidebarMediaItem } from './sidebar-media-item.component';
import { useKeyboardNavigation } from './use-keyboard-navigation.hook';

const layoutOptions = {
    maxColumns: 1,
    minSpace: new Size(8, 8),
    minItemSize: new Size(120, 120),
    maxItemSize: new Size(120, 120),
    preserveAspectRatio: true,
};

type SidebarItemsProps = {
    items: DatasetItem[];
    hasNextPage: boolean;
    isFetchingNextPage: boolean;
    mediaItem: DatasetItem;
    fetchNextPage: () => void;
    onSelectedMediaItem: (item: DatasetItem) => void;
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
    const { mediaState } = useSelectedData();

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
                mediaState={mediaState}
                selectedKeys={new Set([String(mediaItem.id)])}
                layoutOptions={layoutOptions}
                isLoadingMore={isFetchingNextPage}
                scrollToIndex={selectedIndex}
                onLoadMore={() => hasNextPage && fetchNextPage()}
                contentItem={(item) => (
                    <SidebarMediaItem
                        item={item}
                        isSelected={mediaItem.id === item.id}
                        onSelectedMediaItem={onSelectedMediaItem}
                    />
                )}
            />
        </View>
    );
};
