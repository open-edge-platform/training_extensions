// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { DatasetStatisticsView, type DatasetItem } from '../src/constants/shared-types';

export const getMockedDatasetItem = (overrides: Partial<DatasetItem>): DatasetItem => ({
    id: '1',
    subset: 'unassigned',
    user_reviewed: false,
    ...overrides,
});

export const getMockedDatasetStatistics = (overrides: Partial<DatasetStatisticsView> = {}): DatasetStatisticsView => ({
    media_counts: { images: 10, videos: 2, video_frames: 0 },
    annotations_counts: {
        annotated_images: 5,
        annotated_videos: 1,
        annotated_video_frames: 0,
        instances: 6,
        instances_per_label: [],
    },
    ...overrides,
});
