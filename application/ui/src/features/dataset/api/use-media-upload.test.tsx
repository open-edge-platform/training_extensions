// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, screen } from '@testing-library/react';
import { getMockedMediaImage } from 'mocks/mock-media';
import { HttpResponse } from 'msw';
import { renderHook } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { useMediaUpload } from './use-media-upload';

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
        server.use(
            http.post('/api/projects/{project_id}/dataset/media', async ({ request }) => {
                const formData = await request.formData();
                const file = formData.get('file') as File;

                if (file.name === 'broken.jpg') {
                    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                    // @ts-expect-error
                    return HttpResponse.json({ detail: 'Upload failed' }, { status: 500 });
                }

                return HttpResponse.json(getMockedMediaImage({ id: crypto.randomUUID() }), { status: 201 });
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
});
