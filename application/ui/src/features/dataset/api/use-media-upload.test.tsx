// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, waitFor } from '@testing-library/react';
import { getMockedMediaImage } from 'mocks/mock-media';
import { HttpResponse } from 'msw';
import { renderHook } from 'test-utils/render';
import { v4 as uuid } from 'uuid';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { MEDIA_UPLOAD_CONCURRENCY, useMediaUpload } from './use-media-upload';

describe('useMediaUpload', () => {
    it('uploads all selected files', async () => {
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

        act(() => {
            result.current.uploadMedia(files);
        });

        await waitFor(() => {
            expect(result.current.uploadProgress.isUploading).toBe(false);
        });
        expect(uploadedFileNames).toEqual(['image-1.jpg', 'image-2.jpg']);
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

        act(() => {
            result.current.uploadMedia(mockFiles);
        });

        await waitFor(() => {
            expect(result.current.uploadProgress.isUploading).toBe(false);
        });

        expect(maxRunningUploads).toBeLessThanOrEqual(MEDIA_UPLOAD_CONCURRENCY);
        expect(result.current.uploadProgress.completed).toBe(12);
    });

    it('tracks upload progress counters', async () => {
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

        act(() => {
            result.current.uploadMedia(files);
        });

        await waitFor(() => {
            expect(result.current.uploadProgress.isUploading).toBe(false);
        });

        expect(result.current.uploadProgress).toEqual({
            total: 2,
            completed: 2,
            succeeded: 1,
            failed: 1,
            isUploading: false,
        });
        expect(result.current.uploadProgress.isUploading).toBe(false);
    });

    it('queues a new upload call while another upload is in progress', async () => {
        const uploadedFileNames: string[] = [];

        server.use(
            http.post('/api/projects/{project_id}/dataset/media', async ({ request }) => {
                const formData = await request.formData();
                const file = formData.get('file') as File;

                uploadedFileNames.push(file.name);

                await new Promise((resolve) => {
                    setTimeout(resolve, 30);
                });

                return HttpResponse.json(getMockedMediaImage({ id: uuid() }), { status: 201 });
            })
        );

        const { result } = renderHook(() => useMediaUpload());

        const firstUploadFiles = [
            new File(['first-1'], 'first-1.jpg', { type: 'image/jpeg' }),
            new File(['first-2'], 'first-2.jpg', { type: 'image/jpeg' }),
        ];
        const secondUploadFiles = [new File(['second-1'], 'second-1.jpg', { type: 'image/jpeg' })];

        act(() => {
            result.current.uploadMedia(firstUploadFiles);
            result.current.uploadMedia(secondUploadFiles);
        });

        await waitFor(() => {
            expect(uploadedFileNames).toEqual(['first-1.jpg', 'first-2.jpg', 'second-1.jpg']);
        });
        await waitFor(() => {
            expect(result.current.uploadProgress.isUploading).toBe(false);
        });

        expect(result.current.uploadProgress.total).toBe(1);
        expect(result.current.uploadProgress.isUploading).toBe(false);
    });
});
