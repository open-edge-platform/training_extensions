// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { View } from '@geti/ui';

import { Gallery } from './gallery/gallery.component';
import { Toolbar } from './toolbar/toolbar.component';

export const DataCollection = () => {
    return (
        <View padding={'size-350'}>
            <Toolbar />

            <Gallery />
        </View>
    );
};
