// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedAnnotation } from 'mocks/mock-annotation';
import { getMockedLabel } from 'mocks/mock-labels';
import { getMockedMediaImage, getMockedVideoFrame, getMultipleMockedMediaImage } from 'mocks/mock-media';
import { describe, expect, it } from 'vitest';

import { getNextMediaItem, toggleLabel } from './util';

describe('secondary toolbar utils', () => {
    describe('toggleLabel', () => {
        const mockLabel1 = getMockedLabel({ id: 'label-1', name: 'Label 1' });
        const mockLabel2 = getMockedLabel({ id: 'label-2', name: 'Label 2' });
        const mockLabel3 = getMockedLabel({ id: 'label-3', name: 'Label 3' });

        it('add label when it does not exist in annotation', () => {
            const annotation = getMockedAnnotation({
                labels: [mockLabel1, mockLabel2],
            });

            const result = toggleLabel(mockLabel3, annotation.labels);

            expect(result).toEqual([mockLabel1, mockLabel2, mockLabel3]);
        });

        it('remove label when it exists in annotation', () => {
            const annotation = getMockedAnnotation({
                labels: [mockLabel1, mockLabel2, mockLabel3],
            });

            const result = toggleLabel(mockLabel2, annotation.labels);

            expect(result).toEqual([mockLabel1, mockLabel3]);
        });

        it('add label to empty labels array', () => {
            const annotation = getMockedAnnotation({ labels: [] });

            const result = toggleLabel(mockLabel1, annotation.labels);

            expect(result).toEqual([mockLabel1]);
        });

        it('remove the only label from annotation', () => {
            const annotation = getMockedAnnotation({
                labels: [mockLabel1],
            });

            const result = toggleLabel(mockLabel1, annotation.labels);

            expect(result).toEqual([]);
        });
    });

    describe('getNextMediaItem', () => {
        describe('image media items', () => {
            it('returns the next image in the list', () => {
                const items = getMultipleMockedMediaImage(3);
                const result = getNextMediaItem(items[0], items, 1);
                expect(result).toEqual(items[1]);
            });

            it('returns the last item when current item is the last one', () => {
                const items = getMultipleMockedMediaImage(3);
                const result = getNextMediaItem(items[2], items, 1);
                expect(result).toEqual(items[2]);
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

            it('stays on the same video when it is the last media item and already at the last frame', () => {
                const frame = getMockedVideoFrame({ id: 'video-1', frame_number: 9, frame_count: 10 });
                const result = getNextMediaItem(frame, [frame], 1);
                expect(result).toEqual(frame);
            });

            it('advances to the next media item when at the last frame', () => {
                const frame = getMockedVideoFrame({ id: 'video-1', frame_number: 9, frame_count: 10 });
                const nextImage = getMockedMediaImage({ id: 'image-1' });
                const result = getNextMediaItem(frame, [frame, nextImage], 3);
                expect(result).toEqual(nextImage);
            });
        });
    });
});
