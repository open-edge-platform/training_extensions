// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ViewModes } from '@geti/ui';
import { fireEvent, screen, waitFor, waitForElementToBeRemoved } from '@testing-library/react';
import { getMockedDatasetStatistics } from 'mocks/mock-dataset-item';
import { getMockedMediaImage } from 'mocks/mock-media';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../api/utils';
import type { Media } from '../../../../constants/shared-types';
import { server } from '../../../../msw-node-setup';
import { isImage } from '../../../../shared/media-item-utils';
import { SelectedDataProvider } from '../../providers/selected-data-provider.component';
import { Toolbar } from './toolbar.component';

const uploadMediaMock = vi.fn();
const onSelectedMediaItemChangeMock = vi.fn();

vi.mock('../../api/use-media-upload', () => ({
    useMediaUpload: () => ({
        uploadMedia: uploadMediaMock,
        uploadProgress: {
            total: 0,
            completed: 0,
            succeeded: 0,
            failed: 0,
            isUploading: false,
        },
    }),
}));

vi.mock('../hooks/use-select-dataset-item.hook', () => ({
    useSelectDatasetItem: () => ({
        onSelectedMediaItemChange: onSelectedMediaItemChangeMock,
        selectedMediaItem: null,
    }),
}));

vi.mock('../../../models/train-model/train-model.component', () => ({
    TrainModel: () => <div>Train model</div>,
}));

vi.mock('../../import-export/import-export.component', () => ({
    ImportExport: () => <div>ImportExport</div>,
}));

vi.mock('hooks/use-project-identifier.hook', () => ({
    useProjectIdentifier: () => 'project-123',
}));

describe('Toolbar', () => {
    const renderToolbar = async (items: Media[] = []) => {
        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(getMockedProject({ id: 'project-123' }));
            }),
            http.get('/api/projects/{project_id}/dataset/statistics', () => {
                return HttpResponse.json(
                    getMockedDatasetStatistics({
                        media_counts: {
                            images: items.filter(isImage).length,
                            videos: items.filter((item) => !isImage(item)).length,
                            video_frames: 0,
                        },
                    })
                );
            }),
            http.get('/api/projects/{project_id}/dataset/media', () => {
                return HttpResponse.json({
                    items: [getMockedMediaImage({})],
                    pagination: { offset: 0, limit: 1, count: items.length, total: items.length },
                });
            }),
            http.get('/api/projects/{project_id}/dataset/items', () => {
                return HttpResponse.json({
                    pagination: {
                        total: items.length,
                        offset: 0,
                        limit: 0,
                        count: items.length,
                    },
                    items: [],
                });
            })
        );

        const result = render(
            <SelectedDataProvider>
                <Toolbar items={items} viewMode={ViewModes.LARGE} setViewMode={vi.fn()} />
            </SelectedDataProvider>
        );

        await waitForElementToBeRemoved(screen.getByRole('progressbar'));

        return result;
    };

    beforeEach(() => {
        uploadMediaMock.mockClear();
        onSelectedMediaItemChangeMock.mockClear();
    });

    it('delegates selected files to useMediaUpload', async () => {
        const file = new File(['file-content'], 'media-item.jpg', { type: 'image/jpeg' });

        await renderToolbar();

        const input = screen.getByLabelText(/Upload media files/);
        fireEvent.change(input, { target: { files: [file] } });

        expect(uploadMediaMock).toHaveBeenCalledWith([file]);
    });

    it('disables annotate button when there are no items', async () => {
        await renderToolbar();

        expect(screen.getByRole('button', { name: 'Annotate' })).toBeDisabled();
    });

    it('calls onSelectedMediaItemChange with first item when annotate is clicked', async () => {
        const firstItem = getMockedMediaImage({ id: 'first-item' });
        const secondItem = getMockedMediaImage({ id: 'second-item' });

        await renderToolbar([firstItem, secondItem]);

        fireEvent.click(screen.getByRole('button', { name: 'Annotate' }));

        expect(onSelectedMediaItemChangeMock).toHaveBeenCalledWith(firstItem);
    });

    it('selects all items and updates selected count', async () => {
        await renderToolbar([getMockedMediaImage({ id: '1' }), getMockedMediaImage({ id: '2' })]);

        fireEvent.click(screen.getByLabelText('select all'));

        expect(screen.getByText('2 selected')).toBeVisible();
        expect(screen.getByLabelText(/delete media item/i)).toBeVisible();
    });

    it('opens dataset statistics modal when clicking the statistics button', async () => {
        renderToolbar();

        const statsButton = await screen.findByLabelText('dataset statistics');
        fireEvent.click(statsButton);

        await waitFor(() => {
            expect(screen.getByText('Dataset Statistics')).toBeVisible();
            expect(screen.getByText('Number of media')).toBeVisible();
            expect(screen.getByText('Annotated images')).toBeVisible();
        });
    });
});
