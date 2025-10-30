// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { DatasetItem } from 'src/constants/shared-types';

const getMockedMediaItem = (data: Partial<DatasetItem>): DatasetItem => ({
    id: '2f3c9f61-7aa0-4529-a924-193761a64b22',
    name: 'IMG_20210209_161319',
    format: 'png',
    width: 3456,
    height: 4608,
    size: 3586217,
    source_id: null,
    subset: 'unassigned',
    ...data,
});

export const getMultipleMockedMediaItems = (count: number, prefixId = '1'): DatasetItem[] => {
    return Array.from({ length: count }, (_, index) =>
        getMockedMediaItem({
            id: `${prefixId}-item-${index + 1}`,
            name: `${prefixId}-Item ${index + 1}`,
        })
    );
};
