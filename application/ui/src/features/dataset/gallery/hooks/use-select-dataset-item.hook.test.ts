// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act } from '@testing-library/react';
import { getMockedMediaImage, getMockedVideo, getMockedVideoFrame } from 'mocks/mock-media';
import { renderHook } from 'test-utils/render';

import { paths } from '../../../../constants/paths';
import { Media } from '../../../../constants/shared-types';
import { useGetDatasetMediaItems } from '../../../../hooks/use-get-dataset-media-items.hook';
import { useSelectDatasetItem } from './use-select-dataset-item.hook';

const mockNavigate = vi.fn();

vi.mock('react-router', async (importOriginal) => {
    const actual = await importOriginal<typeof import('react-router')>();
    return {
        ...actual,
        useNavigate: () => mockNavigate,
    };
});

const mockDatasetMediaItems = {
    items: [] as Media[],
    fetchNextPage: vi.fn(),
    hasNextPage: false,
    isFetchingNextPage: false,
    isPending: false,
    totalCount: 0,
};

vi.mock('../../../../hooks/use-get-dataset-media-items.hook', () => ({
    useGetDatasetMediaItems: vi.fn(() => mockDatasetMediaItems),
}));

const MOCKED_PROJECT_ID = '123';

vi.mock('../../../../hooks/use-project-identifier.hook', () => ({
    useProjectIdentifier: vi.fn(() => MOCKED_PROJECT_ID),
}));

const SEARCH = '?annotationStatusFilter=with_annotations';

describe('useSelectDatasetItem', () => {
    beforeEach(() => {
        mockNavigate.mockClear();
        vi.mocked(useGetDatasetMediaItems).mockReturnValue({ ...mockDatasetMediaItems, items: [] });
    });

    describe('onSelectedMediaItemChange', () => {
        it('navigates to dataset index with preserved search when item is null', () => {
            const route = `${paths.project.dataset.index({ projectId: MOCKED_PROJECT_ID })}${SEARCH}`;
            const { result } = renderHook(() => useSelectDatasetItem(), {
                route,
                path: paths.project.dataset.index.pattern,
            });

            act(() => {
                result.current.onSelectedMediaItemChange(null);
            });

            expect(mockNavigate).toHaveBeenCalledWith({
                pathname: paths.project.dataset.index({ projectId: MOCKED_PROJECT_ID }),
                search: SEARCH,
            });
        });

        it('navigates to frame 0 with preserved search for a video item', () => {
            const video = getMockedVideo({ id: 'video-42' });
            const route = `${paths.project.dataset.index({ projectId: MOCKED_PROJECT_ID })}${SEARCH}`;
            const { result } = renderHook(() => useSelectDatasetItem(), {
                route,
                path: paths.project.dataset.index.pattern,
            });

            act(() => {
                result.current.onSelectedMediaItemChange(video);
            });

            expect(mockNavigate).toHaveBeenCalledWith({
                pathname: paths.project.dataset.item.frame({
                    datasetItemId: video.id,
                    frameNumber: '0',
                    projectId: MOCKED_PROJECT_ID,
                }),
                search: SEARCH,
            });
        });

        it('navigates to the correct frame number with preserved search for a video frame item', () => {
            const videoFrame = getMockedVideoFrame({ id: 'vf-7', frame_number: 42 });
            const route = `${paths.project.dataset.index({ projectId: MOCKED_PROJECT_ID })}${SEARCH}`;
            const { result } = renderHook(() => useSelectDatasetItem(), {
                route,
                path: paths.project.dataset.index.pattern,
            });

            act(() => {
                result.current.onSelectedMediaItemChange(videoFrame);
            });

            expect(mockNavigate).toHaveBeenCalledWith({
                pathname: paths.project.dataset.item.frame({
                    datasetItemId: videoFrame.id,
                    frameNumber: videoFrame.frame_number.toString(),
                    projectId: MOCKED_PROJECT_ID,
                }),
                search: SEARCH,
            });
        });

        it('navigates to item index with preserved search for an image item', () => {
            const image = getMockedMediaImage({ id: 'img-99' });
            const route = `${paths.project.dataset.index({ projectId: MOCKED_PROJECT_ID })}${SEARCH}`;
            const { result } = renderHook(() => useSelectDatasetItem(), {
                route,
                path: paths.project.dataset.index.pattern,
            });

            act(() => {
                result.current.onSelectedMediaItemChange(image);
            });

            expect(mockNavigate).toHaveBeenCalledWith({
                pathname: paths.project.dataset.item.index({
                    datasetItemId: image.id,
                    projectId: MOCKED_PROJECT_ID,
                }),
                search: SEARCH,
            });
        });

        it('preserves an empty search string when there are no query params', () => {
            const image = getMockedMediaImage({ id: 'img-1' });
            const route = paths.project.dataset.index({ projectId: MOCKED_PROJECT_ID });

            const { result } = renderHook(() => useSelectDatasetItem(), {
                route,
                path: paths.project.dataset.index.pattern,
            });

            act(() => {
                result.current.onSelectedMediaItemChange(image);
            });

            expect(mockNavigate).toHaveBeenCalledWith({
                pathname: paths.project.dataset.item.index({
                    datasetItemId: image.id,
                    projectId: MOCKED_PROJECT_ID,
                }),
                search: '',
            });
        });
    });

    describe('selectedMediaItem', () => {
        it('returns the matching item when datasetItemId param matches an item in the list', () => {
            const image = getMockedMediaImage({ id: 'img-selected' });
            vi.mocked(useGetDatasetMediaItems).mockReturnValue({ ...mockDatasetMediaItems, items: [image] });

            const route = `${paths.project.dataset.item.index({ projectId: MOCKED_PROJECT_ID, datasetItemId: image.id })}${SEARCH}`;
            const { result } = renderHook(() => useSelectDatasetItem(), {
                route,
                path: paths.project.dataset.item.index.pattern,
            });

            expect(result.current.selectedMediaItem).toEqual(image);
        });

        it('returns null when datasetItemId does not match any item', () => {
            const image = getMockedMediaImage({ id: 'img-other' });
            vi.mocked(useGetDatasetMediaItems).mockReturnValue({ ...mockDatasetMediaItems, items: [image] });

            const route = `${paths.project.dataset.item.index({ projectId: MOCKED_PROJECT_ID, datasetItemId: `${image.id}-23` })}${SEARCH}`;
            const { result } = renderHook(() => useSelectDatasetItem(), {
                route,
                path: paths.project.dataset.item.index.pattern,
            });

            expect(result.current.selectedMediaItem).toBeNull();
        });

        it('returns null when there are no items', () => {
            vi.mocked(useGetDatasetMediaItems).mockReturnValue({ ...mockDatasetMediaItems, items: [] });

            const route = `${paths.project.dataset.item.index({ projectId: MOCKED_PROJECT_ID, datasetItemId: `img-selected` })}${SEARCH}`;
            const { result } = renderHook(() => useSelectDatasetItem(), {
                route,
                path: paths.project.dataset.item.index.pattern,
            });

            expect(result.current.selectedMediaItem).toBeNull();
        });
    });
});
