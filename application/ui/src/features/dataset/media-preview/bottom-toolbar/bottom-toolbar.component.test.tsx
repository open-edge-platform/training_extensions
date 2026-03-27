// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen, waitFor } from '@testing-library/react';
import { getMockedDatasetItem } from 'mocks/mock-dataset-item';
import { getMockedMediaImage } from 'mocks/mock-media';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { ZoomProvider } from '../../../../components/zoom/zoom.provider';
import type { MediaImage } from '../../../../constants/shared-types';
import { server } from '../../../../msw-node-setup';
import { AnnotationVisibilityProvider } from '../../../../shared/annotator/annotation-visibility-provider.component';
import { CanvasSettingsProvider } from '../primary-toolbar/settings/canvas-settings-provider.component';
import { BottomToolbar } from './bottom-toolbar.component';

type BottomToolbarProps = {
    mediaItem: MediaImage;
};

const renderBottomToolbar = ({ mediaItem }: BottomToolbarProps) => {
    return render(
        <ZoomProvider>
            <AnnotationVisibilityProvider>
                <CanvasSettingsProvider>
                    <BottomToolbar mediaItem={mediaItem} />
                </CanvasSettingsProvider>
            </AnnotationVisibilityProvider>
        </ZoomProvider>
    );
};

describe('BottomToolbar', () => {
    const mockMediaItem = getMockedMediaImage({
        id: 'media-123',
        name: 'test-image',
        format: 'jpg',
        width: 1920,
        height: 1080,
    });

    it('displays the filename with correct format and dimensions', () => {
        renderBottomToolbar({ mediaItem: mockMediaItem });

        expect(screen.getByText('test-image.jpg (1920 x 1080 px)')).toBeInTheDocument();
    });

    it('displays "Accepted" tag when user has reviewed the media', async () => {
        server.use(
            http.get('/api/projects/{project_id}/dataset/items/{dataset_item_id}', () => {
                return HttpResponse.json(getMockedDatasetItem({ id: 'media-123', user_reviewed: true }), {
                    status: 200,
                });
            })
        );

        renderBottomToolbar({ mediaItem: mockMediaItem });

        expect(await screen.findByLabelText('Accepted')).toBeInTheDocument();
    });

    it('displays "For Review" tag when user has not reviewed the media', () => {
        server.use(
            http.get('/api/projects/{project_id}/dataset/items/{dataset_item_id}', () => {
                return HttpResponse.json(getMockedDatasetItem({ id: 'media-123', user_reviewed: false }), {
                    status: 200,
                });
            })
        );

        renderBottomToolbar({ mediaItem: mockMediaItem });

        expect(screen.getByLabelText('For Review')).toBeInTheDocument();
    });

    it('renders the subset picker with default placeholder', async () => {
        server.use(
            http.get('/api/projects/{project_id}/dataset/items/{dataset_item_id}', () => {
                return HttpResponse.json(
                    getMockedDatasetItem({ id: 'media-123', user_reviewed: false, subset: 'unassigned' }),
                    {
                        status: 200,
                    }
                );
            })
        );

        renderBottomToolbar({ mediaItem: mockMediaItem });

        expect(await screen.findByLabelText('Select subset')).toBeInTheDocument();
    });

    it('renders the subset instead of picker when subset is assigned', async () => {
        server.use(
            http.get('/api/projects/{project_id}/dataset/items/{dataset_item_id}', () => {
                return HttpResponse.json(
                    getMockedDatasetItem({ id: 'media-123', user_reviewed: false, subset: 'validation' }),
                    {
                        status: 200,
                    }
                );
            })
        );

        renderBottomToolbar({ mediaItem: mockMediaItem });

        expect(await screen.findByLabelText('Validation')).toBeInTheDocument();
    });

    it('calls annotations endpoint when changing subset', async () => {
        const postSpy = vi.fn();

        server.use(
            http.get('/api/projects/{project_id}/dataset/items/{dataset_item_id}', () => {
                return HttpResponse.json(getMockedDatasetItem({ id: 'media-123', subset: 'unassigned' }), {
                    status: 200,
                });
            }),
            http.get('/api/projects/{project_id}/dataset/media/{media_id}/annotations', () => {
                return HttpResponse.json({ annotations: [], user_reviewed: false }, { status: 200 });
            }),
            http.post(
                '/api/projects/{project_id}/dataset/media/{media_id}/annotations',
                async ({ request, params }) => {
                    const body = await request.json();

                    postSpy(params, body);

                    return HttpResponse.json({
                        media_id: 'media-123',
                        annotations: [],
                        prediction_model_id: null,
                        user_reviewed: false,
                    });
                }
            )
        );

        renderBottomToolbar({ mediaItem: mockMediaItem });

        const pickerButton = await screen.findByRole('button', { name: /select subset/i });
        fireEvent.click(pickerButton);

        const validationOption = await screen.findByRole('option', { name: /Validation/i });
        fireEvent.click(validationOption);

        await waitFor(() => {
            expect(postSpy).toHaveBeenCalledWith(
                expect.objectContaining({
                    project_id: expect.any(String),
                    media_id: 'media-123',
                }),
                expect.objectContaining({
                    annotations: [],
                    subset: 'validation',
                })
            );
        });
    });
});
