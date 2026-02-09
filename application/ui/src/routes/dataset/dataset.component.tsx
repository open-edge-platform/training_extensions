// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Flex } from '@geti/ui';
import { useGetDatasetMediaItems } from 'hooks/use-get-dataset-media-items.hook';

import { Gallery } from '../../features/dataset/gallery/gallery.component';
import { Toolbar } from '../../features/dataset/gallery/toolbar/toolbar.component';

export const Dataset = () => {
    const { items, hasNextPage, isFetchingNextPage, fetchNextPage } = useGetDatasetMediaItems();

    return (
        <Flex
            height={'100%'}
            gridArea={'content'}
            direction={'column'}
            UNSAFE_style={{ padding: dimensionValue('size-350'), paddingBottom: 0 }}
        >
            <Toolbar items={items} />

            <Gallery
                items={items}
                fetchNextPage={fetchNextPage}
                hasNextPage={hasNextPage}
                isFetchingNextPage={isFetchingNextPage}
            />
        </Flex>
    );
};
