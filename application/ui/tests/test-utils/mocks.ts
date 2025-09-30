// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Project } from 'src/features/project/types';

import { DatasetItem } from '../../src/features/annotator/types';

export const getMockedMediaItem = (data: Partial<DatasetItem>): DatasetItem => ({
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

export const getMockedProject = (customProject: Partial<Project>): Project => {
    return {
        id: '7b073838-99d3-42ff-9018-4e901eb047fc',
        name: 'animals',
        task: {
            exclusive_labels: true,
            labels: [
                {
                    color: '#FF5733',
                    hotkey: 'S',
                    id: 'a22d82ba-afa9-4d6e-bbc1-8c8e4002ec29',
                    name: 'Object',
                },
            ],
            task_type: 'classification',
        },
        ...customProject,
    };
};
