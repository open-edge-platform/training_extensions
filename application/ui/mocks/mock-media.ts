// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { MediaImage, MediaVideo, MediaVideoFrame } from '../src/constants/shared-types';

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

export const getMockedVideo = (props: Partial<MediaVideo> = {}): MediaVideo => ({
    id: 'video-1',
    type: 'video',
    fps: 60,
    duration: 123,
    width: 400,
    height: 400,
    size: 123,
    name: 'video',
    format: 'mp4',
    frame_count: 400,
    annotated_frame_count: 100,
    source_id: undefined,
    ...props,
});

export const getMockedVideoFrame = (props: Partial<MediaVideoFrame> = {}): MediaVideoFrame => ({
    id: 'video-1',
    type: 'video_frame',
    name: 'video-1.mp4',
    format: 'mp4',
    width: 400,
    height: 400,
    size: 0,
    fps: 60,
    duration: 10,
    frame_count: 10,
    annotated_frame_count: 5,
    frame_stride: 1,
    frame_number: 0,
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
