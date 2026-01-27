// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef } from 'react';

import { Size, useUnwrapDOMRef, View } from '@geti/ui';

import { VirtualizerGridLayout } from '../../../../components/virtualizer-grid-layout/virtualizer-grid-layout.component';
import type { Media } from '../../../../constants/shared-types';
import { useSelectedData } from '../../selected-data-provider.component';
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
                    <SidebarMediaItem item={item} mediaState={mediaState} onSelectedMediaItem={onSelectedMediaItem} />
                )}
            />
        </View>
    );
};
