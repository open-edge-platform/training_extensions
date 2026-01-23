// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Media } from '../src/constants/shared-types';

export const mockedMedia = (props: Partial<Media> = {}): Media => ({
    id: 'item-1',
    type: 'image',
    name: 'item-1.jpg',
    format: 'jpg',
    width: 0,
    height: 0,
    size: 0,
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
