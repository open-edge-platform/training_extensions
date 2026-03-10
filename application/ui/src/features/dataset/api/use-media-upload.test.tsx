// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, screen } from '@testing-library/react';
import { getMockedMediaImage } from 'mocks/mock-media';
import { HttpResponse } from 'msw';
import { renderHook } from 'test-utils/render';
import { v4 as uuid } from 'uuid';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { MEDIA_UPLOAD_CONCURRENCY, useMediaUpload } from './use-media-upload';

describe('useMediaUpload', () => {
    it('uploads all selected files and shows a success toast', async () => {
        const uploadedFileNames: string[] = [];

        server.use(
            http.post('/api/projects/{project_id}/dataset/media', async ({ request, params }) => {
                const formData = await request.formData();
                const file = formData.get('file');

                uploadedFileNames.push((file as File).name);
                expect(params.project_id).toBe('123');

                return HttpResponse.json(getMockedMediaImage({ id: crypto.randomUUID() }), { status: 201 });
            })
        );

        const { result } = renderHook(() => useMediaUpload());

        const files = [
            new File(['file-1'], 'image-1.jpg', { type: 'image/jpeg' }),
            new File(['file-2'], 'image-2.jpg', { type: 'image/jpeg' }),
        ];

        await act(async () => {
            await result.current.uploadMedia(files);
        });

        expect(uploadedFileNames).toEqual(['image-1.jpg', 'image-2.jpg']);
        expect(await screen.findByText('Uploaded 2 item(s)')).toBeVisible();
    });

    it('shows a warning toast when some uploads fail', async () => {
        let requestCount = 0;

        server.use(
            http.post('/api/projects/{project_id}/dataset/media', async () => {
                requestCount += 1;

                if (requestCount === 2) {
                    return HttpResponse.error();
                }

                return HttpResponse.json(getMockedMediaImage({ id: uuid() }), { status: 201 });
            })
        );

        const { result } = renderHook(() => useMediaUpload());

        const files = [
            new File(['ok-file'], 'ok.jpg', { type: 'image/jpeg' }),
            new File(['broken-file'], 'broken.jpg', { type: 'image/jpeg' }),
        ];

        await act(async () => {
            await result.current.uploadMedia(files);
        });

        expect(await screen.findByText('Uploaded 1 item(s), 1 failed')).toBeVisible();
    });

    it('shows an error toast when all uploads fail', async () => {
        server.use(
            http.post('/api/projects/{project_id}/dataset/media', async () => {
                return HttpResponse.error();
            })
        );

        const { result } = renderHook(() => useMediaUpload());

        const files = [
            new File(['broken-file-1'], 'broken-1.jpg', { type: 'image/jpeg' }),
            new File(['broken-file-2'], 'broken-2.jpg', { type: 'image/jpeg' }),
        ];

        await act(async () => {
            await result.current.uploadMedia(files);
        });

        expect(await screen.findByText('Failed to upload 2 item(s)')).toBeVisible();
    });

    it('does not exceed configured upload concurrency', async () => {
        let runningUploads = 0;
        let maxRunningUploads = 0;

        server.use(
            http.post('/api/projects/{project_id}/dataset/media', async () => {
                runningUploads += 1;
                maxRunningUploads = Math.max(maxRunningUploads, runningUploads);

                await new Promise((resolve) => {
                    setTimeout(resolve, 20);
                });

                runningUploads -= 1;

                return HttpResponse.json(getMockedMediaImage({ id: uuid() }), { status: 201 });
            })
        );

        const { result } = renderHook(() => useMediaUpload());

        const mockFiles = Array.from(
            { length: 12 },
            (_, index) => new File([`file-${index}`], `image-${index}.jpg`, { type: 'image/jpeg' })
        );

        await act(async () => {
            await result.current.uploadMedia(mockFiles);
        });

        expect(maxRunningUploads).toBeLessThanOrEqual(MEDIA_UPLOAD_CONCURRENCY);
        expect(await screen.findByText('Uploaded 12 item(s)')).toBeVisible();
    });
});
