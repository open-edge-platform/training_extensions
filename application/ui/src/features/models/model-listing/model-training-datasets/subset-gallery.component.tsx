// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Loading, Size, Text, View } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { MediaItem } from '../../../../components/media-item/media-item.component';
import { MediaThumbnail } from '../../../../components/media-thumbnail/media-thumbnail.component';
import { VirtualizerGridLayout } from '../../../../components/virtualizer-grid-layout/virtualizer-grid-layout.component';
import type { DatasetItem } from '../../../../constants/shared-types';
import { getThumbnailUrl } from '../../../../shared/media-url.utils';

const layoutOptions = {
    minSpace: new Size(4, 4),
    minItemSize: new Size(80, 80),
    maxColumns: 4,
    preserveAspectRatio: true,
};

type SubsetGalleryProps = {
    items: DatasetItem[];
    fetchNextPage: () => void;
    hasNextPage: boolean;
    isFetchingNextPage: boolean;
    isLoading: boolean;
};

export const SubsetGallery = ({
    items,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
    fetchNextPage,
}: SubsetGalleryProps) => {
    const projectId = useProjectIdentifier();

    if (isLoading) {
        return (
            <Flex height={'100%'}>
                <Loading />
            </Flex>
        );
    }

    if (items.length === 0) {
        return (
            <Flex height={'100%'}>
                <Text>No items in this subset</Text>
            </Flex>
        );
    }

    return (
        <View height={'100%'} width={'100%'} position={'relative'} minHeight={'size-5000'}>
            <VirtualizerGridLayout
                items={items}
                ariaLabel='subset-media-grid'
                selectionMode='none'
                layoutOptions={layoutOptions}
                isLoadingMore={isFetchingNextPage}
                onLoadMore={() => hasNextPage && fetchNextPage()}
                contentItem={(item) => (
                    <MediaItem
                        contentElement={() => (
                            <MediaThumbnail
                                alt={item.name}
                                url={getThumbnailUrl(projectId, item.id)}
                                // TODO: leverage onDoubleClick to open a dialog
                            />
                        )}
                    />
                )}
            />
        </View>
    );
};
