// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Media } from '../src/constants/shared-types';

const getMockedMediaItem = (data: Partial<Media>): Media => ({
    id: '2f3c9f61-7aa0-4529-a924-193761a64b22',
    type: 'image',
    name: 'IMG_20210209_161319',
    format: 'png',
    width: 3456,
    height: 4608,
    size: 3586217,
    ...data,
});

export const getMultipleMockedMediaItems = (count: number, prefixId = '1'): Media[] => {
    return Array.from({ length: count }, (_, index) =>
        getMockedMediaItem({
            id: `${prefixId}-item-${index + 1}`,
            name: `${prefixId}-Item ${index + 1}`,
        })
    );
};
