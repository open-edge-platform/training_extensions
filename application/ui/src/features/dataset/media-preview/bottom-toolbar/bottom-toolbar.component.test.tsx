// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen, waitForElementToBeRemoved } from '@testing-library/react';
import { getMockedDatasetItem } from 'mocks/mock-dataset-item';
import { getMockedMediaImage } from 'mocks/mock-media';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { ZoomProvider } from '../../../../components/zoom/zoom.provider';
import type { MediaImage } from '../../../../constants/shared-types';
import { server } from '../../../../msw-node-setup';
import { AnnotationActionsProvider } from '../../../../shared/annotator/annotation-actions-provider.component';
import { AnnotationVisibilityProvider } from '../../../../shared/annotator/annotation-visibility-provider.component';
import { CanvasSettingsProvider } from '../primary-toolbar/settings/canvas-settings-provider.component';
import { BottomToolbar } from './bottom-toolbar.component';

type BottomToolbarProps = {
    mediaItem: MediaImage;
};

const renderBottomToolbar = async ({ mediaItem }: BottomToolbarProps) => {
    render(
        <ZoomProvider>
            <AnnotationVisibilityProvider>
                <CanvasSettingsProvider>
                    <AnnotationActionsProvider
                        mediaItem={mediaItem}
                        initialAnnotationsDTO={[]}
                        initialPredictionsDTO={[]}
                        isUserReviewed={false}
                        mode={'annotation'}
                    >
                        <BottomToolbar mediaItem={mediaItem} />
                    </AnnotationActionsProvider>
                </CanvasSettingsProvider>
            </AnnotationVisibilityProvider>
        </ZoomProvider>
    );

    await waitForElementToBeRemoved(screen.getByRole('progressbar'));
};

describe('BottomToolbar', () => {
    const mockMediaItem = getMockedMediaImage({
        id: 'media-123',
        name: 'test-image',
        format: 'jpg',
        width: 1920,
        height: 1080,
    });

    beforeEach(() => {
        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(getMockedProject({}));
            })
        );
    });

    it('displays the filename with correct format and dimensions', async () => {
        await renderBottomToolbar({ mediaItem: mockMediaItem });

        expect(await screen.findByText('test-image.jpg (1920 x 1080 px)')).toBeInTheDocument();
    });

    it('displays "Accepted" tag when user has reviewed the media', async () => {
        server.use(
            http.get('/api/projects/{project_id}/dataset/items/{dataset_item_id}', () => {
                return HttpResponse.json(getMockedDatasetItem({ id: 'media-123', user_reviewed: true }), {
                    status: 200,
                });
            })
        );

        await renderBottomToolbar({ mediaItem: mockMediaItem });

        expect(await screen.findByLabelText('Accepted')).toBeInTheDocument();
    });

    it('displays "For Review" tag when user has not reviewed the media', async () => {
        server.use(
            http.get('/api/projects/{project_id}/dataset/items/{dataset_item_id}', () => {
                return HttpResponse.json(getMockedDatasetItem({ id: 'media-123', user_reviewed: false }), {
                    status: 200,
                });
            })
        );

        await renderBottomToolbar({ mediaItem: mockMediaItem });

        expect(await screen.findByLabelText('For Review')).toBeInTheDocument();
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

        await renderBottomToolbar({ mediaItem: mockMediaItem });

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

        await renderBottomToolbar({ mediaItem: mockMediaItem });

        expect(await screen.findByLabelText('Validation')).toBeInTheDocument();
    });

    it('shows pending subset tag after selection', async () => {
        await renderBottomToolbar({ mediaItem: mockMediaItem });

        const pickerButton = await screen.findByRole('button', { name: /select subset/i });
        fireEvent.click(pickerButton);

        const validationOption = screen.getByRole('option', { name: /Validation/i });
        fireEvent.click(validationOption);

        expect(screen.getByLabelText('Validation')).toBeInTheDocument();
    });
});
