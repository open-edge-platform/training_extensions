// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedMediaImage, getMockedVideo } from 'mocks/mock-media';

import { getNumberOfImagesAndVideosMessage, toggleMultipleSelection } from './util';

describe('toggleMultipleSelection', () => {
    const items = ['a', 'b', 'c'];

    it('returns empty set if selectedItems is "all"', () => {
        const result = toggleMultipleSelection(items)('all');
        expect(result).toEqual(new Set());
    });

    it('should select all items if selectedItems is empty set', () => {
        const result = toggleMultipleSelection(items)(new Set());
        expect(result).toEqual(new Set(items));
    });

    it('should select all items if some items are selected', () => {
        const result = toggleMultipleSelection(items)(new Set(['a']));
        expect(result).toEqual(new Set(items));
    });

    it('should deselect all items if all items are selected', () => {
        const result = toggleMultipleSelection(items)(new Set(items));
        expect(result).toEqual(new Set());
    });

    it('should select all items if more than one but not all items are selected', () => {
        const result = toggleMultipleSelection(items)(new Set(['a', 'b']));
        expect(result).toEqual(new Set(items));
    });
});

describe('getNumberOfImagesAndVideosMessage', () => {
    it('returns empty string for an empty array', () => {
        expect(getNumberOfImagesAndVideosMessage([])).toBe('');
    });

    it('returns "1 image" for a single image', () => {
        expect(getNumberOfImagesAndVideosMessage([getMockedMediaImage()])).toBe('1 image');
    });

    it('returns plural "images" for multiple images', () => {
        const images = [
            getMockedMediaImage({ id: '1' }),
            getMockedMediaImage({ id: '2' }),
            getMockedMediaImage({ id: '3' }),
        ];
        expect(getNumberOfImagesAndVideosMessage(images)).toBe('3 images');
    });

    it('returns "1 video" for a single video', () => {
        expect(getNumberOfImagesAndVideosMessage([getMockedVideo()])).toBe('1 video');
    });

    it('returns plural "videos" for multiple videos', () => {
        const videos = [getMockedVideo({ id: 'v1' }), getMockedVideo({ id: 'v2' }), getMockedVideo({ id: 'v3' })];
        expect(getNumberOfImagesAndVideosMessage(videos)).toBe('3 videos');
    });

    it('returns combined message for a mix of images and videos', () => {
        const media = [
            getMockedMediaImage({ id: 'img1' }),
            getMockedMediaImage({ id: 'img2' }),
            getMockedVideo({ id: 'v1' }),
            getMockedVideo({ id: 'v2' }),
        ];
        expect(getNumberOfImagesAndVideosMessage(media)).toBe('2 images, 2 videos');
    });

    it('uses singular "video" (no trailing "s") when exactly 1 video appears in a mixed array', () => {
        const media = [
            getMockedMediaImage({ id: 'img1' }),
            getMockedMediaImage({ id: 'img2' }),
            getMockedVideo({ id: 'v1' }),
        ];
        expect(getNumberOfImagesAndVideosMessage(media)).toBe('2 images, 1 video');
    });
});
