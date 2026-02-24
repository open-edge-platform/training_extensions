// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { MediaImage } from '../src/constants/shared-types';

export const getMockedMediaImage = (props: Partial<MediaImage> = {}): MediaImage => ({
    id: 'item-1',
    type: 'image',
    name: 'item-1.jpg',
    format: 'jpg',
    width: 0,
    height: 0,
    size: 0,
    ...props,
});

export const getMultipleMockedMediaImage = (count: number, prefixId = '1'): MediaImage[] => {
    return Array.from({ length: count }, (_, index) =>
        getMockedMediaImage({
            id: `${prefixId}-item-${index + 1}`,
            name: `${prefixId}-Item ${index + 1}`,
        })
    );
};
