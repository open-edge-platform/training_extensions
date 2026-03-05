// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedMediaImage, getMockedVideoFrame, getMultipleMockedMediaImage } from 'mocks/mock-media';

import type { AnnotationDTO } from '../../../constants/shared-types';
import { getInitialAnnotations, getInitialPredictions, getNextMediaItem } from './utils';

describe('getInitialAnnotations', () => {
    const mockAnnotations: AnnotationDTO[] = [
        {
            shape: { type: 'rectangle', x: 0, y: 0, width: 100, height: 100 },
            labels: [{ id: '1' }],
        },
        {
            shape: {
                type: 'polygon',
                points: [
                    { x: 0, y: 0 },
                    { x: 100, y: 100 },
                ],
            },
            labels: [{ id: '2' }],
        },
        {
            shape: { type: 'full_image' },
            labels: [{ id: '3' }],
        },
    ];

    it('returns annotations when user has reviewed', () => {
        const result = getInitialAnnotations(true, mockAnnotations);
        expect(result).toEqual(mockAnnotations);
    });

    it('returns empty array when user has not reviewed', () => {
        const result = getInitialAnnotations(false, mockAnnotations);
        expect(result).toEqual([]);
    });
});

describe('getInitialPredictions', () => {
    const mockAnnotations: AnnotationDTO[] = [
        {
            shape: { type: 'rectangle', x: 0, y: 0, width: 100, height: 100 },
            labels: [{ id: '1' }],
        },
        {
            shape: {
                type: 'polygon',
                points: [
                    { x: 0, y: 0 },
                    { x: 100, y: 100 },
                ],
            },
            labels: [{ id: '2' }],
        },
        {
            shape: { type: 'full_image' },
            labels: [{ id: '3' }],
        },
    ];

    it('returns empty array when user has reviewed', () => {
        const result = getInitialPredictions(true, mockAnnotations);
        expect(result).toEqual([]);
    });

    it('returns annotations when user has not reviewed', () => {
        const result = getInitialPredictions(false, mockAnnotations);
        expect(result).toEqual(mockAnnotations);
    });
});

describe('getNextMediaItem', () => {
    describe('image media items', () => {
        it('returns the next image in the list', () => {
            const items = getMultipleMockedMediaImage(3);
            const result = getNextMediaItem(items[0], items, 1);
            expect(result).toEqual(items[1]);
        });

        it('returns undefined when current item is the last one', () => {
            const items = getMultipleMockedMediaImage(3);
            const result = getNextMediaItem(items[2], items, 1);
            expect(result).toBeUndefined();
        });

        it('returns the first item when current item is not found in the list', () => {
            const items = getMultipleMockedMediaImage(3);
            const unknownItem = getMockedMediaImage({ id: 'unknown' });
            const result = getNextMediaItem(unknownItem, items, 1);
            expect(result).toEqual(items[0]);
        });
    });

    describe('video frame media items', () => {
        it('returns the next video frame based on step', () => {
            const frame = getMockedVideoFrame({ frame_number: 0, frame_count: 10 });
            const result = getNextMediaItem(frame, [], 1);
            expect(result).toEqual({ ...frame, frame_number: 1 });
        });

        it('advances by the given step size', () => {
            const frame = getMockedVideoFrame({ frame_number: 0, frame_count: 10 });
            const result = getNextMediaItem(frame, [], 3);
            expect(result).toEqual({ ...frame, frame_number: 3 });
        });

        it('advances to the next media item when already at the last video frame', () => {
            const frame = getMockedVideoFrame({ id: 'video-1', frame_number: 9, frame_count: 10 });
            const nextImage = getMockedMediaImage({ id: 'image-1' });
            const result = getNextMediaItem(frame, [frame, nextImage], 1);
            expect(result).toEqual(nextImage);
        });

        it('returns undefined when it is the last media item and already at the last frame', () => {
            const frame = getMockedVideoFrame({ id: 'video-1', frame_number: 9, frame_count: 10 });
            const result = getNextMediaItem(frame, [frame], 1);
            expect(result).toBeUndefined();
        });

        it('advances to the next media item when at the last frame', () => {
            const frame = getMockedVideoFrame({ id: 'video-1', frame_number: 9, frame_count: 10 });
            const nextImage = getMockedMediaImage({ id: 'image-1' });
            const result = getNextMediaItem(frame, [frame, nextImage], 3);
            expect(result).toEqual(nextImage);
        });

        it('handles a frame_number that is not aligned with the step', () => {
            const frame = getMockedVideoFrame({ id: 'video-1', frame_number: 5, frame_count: 10 });
            const result = getNextMediaItem(frame, [frame], 3);
            expect(result).toEqual({ ...frame, frame_number: 6 });
        });
    });
});
