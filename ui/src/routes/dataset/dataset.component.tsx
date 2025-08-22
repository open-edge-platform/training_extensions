// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Flex } from '@geti/ui';

import { Gallery } from '../../features/dataset/gallery/gallery.component';
import { Toolbar } from '../../features/dataset/toolbar/toolbar.component';

export const Dataset = () => {
    return (
        <Flex
            height={'100%'}
            gridArea={'content'}
            direction={'column'}
            UNSAFE_style={{ padding: dimensionValue('size-350'), paddingBottom: 0, boxSizing: 'border-box' }}
        >
            <Toolbar />

            <Gallery />
        </Flex>
    );
};
