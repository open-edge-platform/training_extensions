// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen, waitFor } from '@testing-library/react';
import { getMockedDatasetItem } from 'mocks/mock-dataset-item';
import { mockedMedia } from 'mocks/mock-media';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { ZoomProvider } from '../../../../components/zoom/zoom.provider';
import { server } from '../../../../msw-node-setup';
import { AnnotationVisibilityProvider } from '../../../../shared/annotator/annotation-visibility-provider.component';
import { CanvasSettingsProvider } from '../primary-toolbar/settings/canvas-settings-provider.component';
import { BottomToolbar } from './bottom-toolbar.component';

type BottomToolbarProps = {
    isUserReviewed: boolean;
    mediaItem: ReturnType<typeof mockedMedia>;
};

const renderBottomToolbar = ({ isUserReviewed, mediaItem }: BottomToolbarProps) => {
    return render(
        <ZoomProvider>
            <AnnotationVisibilityProvider>
                <CanvasSettingsProvider>
                    <BottomToolbar isUserReviewed={isUserReviewed} mediaItem={mediaItem} />
                </CanvasSettingsProvider>
            </AnnotationVisibilityProvider>
        </ZoomProvider>
    );
};

describe('BottomToolbar', () => {
    const mockMediaItem = mockedMedia({
        id: 'media-123',
        name: 'test-image',
        format: 'jpg',
        width: 1920,
        height: 1080,
    });

    it('displays the filename with correct format and dimensions', () => {
        renderBottomToolbar({ isUserReviewed: false, mediaItem: mockMediaItem });

        expect(screen.getByText('test-image.jpg (1920 x 1080 px)')).toBeInTheDocument();
    });

    it('displays "Accepted" tag when user has reviewed the media', () => {
        renderBottomToolbar({ isUserReviewed: true, mediaItem: mockMediaItem });

        expect(screen.getByLabelText('Accepted')).toBeInTheDocument();
    });

    it('displays "For Review" tag when user has not reviewed the media', () => {
        renderBottomToolbar({ isUserReviewed: false, mediaItem: mockMediaItem });

        expect(screen.getByLabelText('For Review')).toBeInTheDocument();
    });

    it('renders the subset picker with default placeholder', () => {
        renderBottomToolbar({ isUserReviewed: false, mediaItem: mockMediaItem });

        expect(screen.getByLabelText('Select subset')).toBeInTheDocument();
    });

    it('calls patch mutation when subset is changed', async () => {
        const patchSpy = vi.fn();

        server.use(
            http.get('/api/projects/{project_id}/dataset/items/{dataset_item_id}', () => {
                return HttpResponse.json(getMockedDatasetItem({ id: 'media-123', subset: 'unassigned' }), {
                    status: 200,
                });
            }),
            http.patch(
                '/api/projects/{project_id}/dataset/items/{dataset_item_id}/subset',
                async ({ request, params }) => {
                    const body = await request.json();

                    patchSpy(params, body);

                    return HttpResponse.json(getMockedDatasetItem({ id: 'media-123', subset: 'validation' }), {
                        status: 200,
                    });
                }
            )
        );

        renderBottomToolbar({ isUserReviewed: false, mediaItem: mockMediaItem });

        const pickerButton = screen.getByRole('button', { name: /select subset/i });
        fireEvent.click(pickerButton);

        const validationOption = await screen.findByRole('option', { name: /Validation/i });
        fireEvent.click(validationOption);

        await waitFor(() => {
            expect(patchSpy).toHaveBeenCalledWith(
                expect.objectContaining({
                    project_id: expect.any(String),
                    dataset_item_id: 'media-123',
                }),
                expect.objectContaining({
                    subset: 'validation',
                })
            );
        });
    });
});
