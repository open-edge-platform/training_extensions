// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Media } from '../src/constants/shared-types';

export const mockedMedia = (props: Partial<Media> = {}): Media => ({
    id: 'item-1',
    type: 'image',
    name: 'item-1.jpg',
    format: 'jpg',
    width: 0,
    height: 0,
    size: 0,
    fps: null,
    frame_count: null,
    ...props,
});

export const getMultipleMockedMedia = (count: number, prefixId = '1'): Media[] => {
    return Array.from({ length: count }, (_, index) =>
        mockedMedia({
            id: `${prefixId}-item-${index + 1}`,
            name: `${prefixId}-Item ${index + 1}`,
        })
    );
};
