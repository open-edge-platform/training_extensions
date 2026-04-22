// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitForElementToBeRemoved } from '@testing-library/react';
import { getMockedDatasetStatistics } from 'mocks/mock-dataset-item';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { TotalItems } from './total-items.component';

vi.mock('hooks/use-project-identifier.hook', () => ({
    useProjectIdentifier: () => 'project-123',
}));

describe('TotalItems', () => {
    const renderTotalItems = async (
        totalSelectedElements: number,
        mediaCounts: { images: number; videos: number; video_frames: number } = {
            images: 0,
            videos: 0,
            video_frames: 0,
        }
    ) => {
        server.use(
            http.get('/api/projects/{project_id}/dataset/statistics', () => {
                return HttpResponse.json(getMockedDatasetStatistics({ media_counts: mediaCounts }));
            })
        );

        const result = render(<TotalItems totalSelectedElements={totalSelectedElements} />);

        await waitForElementToBeRemoved(screen.getByRole('progressbar'));

        return result;
    };

    it('shows selected count when items are selected', async () => {
        await renderTotalItems(3, { images: 5, videos: 0, video_frames: 0 });

        expect(screen.getByText('3 selected')).toBeVisible();
    });

    it('shows images and videos when both exist', async () => {
        await renderTotalItems(0, { images: 4, videos: 2, video_frames: 0 });

        expect(screen.getByText('4 images, 2 videos')).toBeVisible();
    });

    it('shows only videos when there are no images', async () => {
        await renderTotalItems(0, { images: 0, videos: 3, video_frames: 0 });

        expect(screen.getByText('3 videos')).toBeVisible();
    });

    it('shows only images when there are no videos', async () => {
        await renderTotalItems(0, { images: 5, videos: 0, video_frames: 0 });

        expect(screen.getByText('5 images')).toBeVisible();
    });

    it('uses singular "image" for exactly one image', async () => {
        await renderTotalItems(0, { images: 1, videos: 0, video_frames: 0 });

        expect(screen.getByText('1 image')).toBeVisible();
    });

    it('uses singular "video" for exactly one video', async () => {
        await renderTotalItems(0, { images: 0, videos: 1, video_frames: 0 });

        expect(screen.getByText('1 video')).toBeVisible();
    });

    it('uses singular forms when both counts are one', async () => {
        await renderTotalItems(0, { images: 1, videos: 1, video_frames: 0 });

        expect(screen.getByText('1 image, 1 video')).toBeVisible();
    });
});
