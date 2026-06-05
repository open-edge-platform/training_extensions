// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, screen } from '@testing-library/react';
import { renderHook } from 'test-utils/render';

import { MediaUploadProvider } from '../providers/media-upload-provider.component';
import { useUploadProgress } from './use-display-upload-progress';

const makeFile = (name: string, size = 100): File => {
    return new File(['x'.repeat(size)], name, { type: 'image/jpeg' });
};

const renderUploadHook = () => renderHook(() => useUploadProgress(), { wrapper: MediaUploadProvider });

describe('useUploadProgress', () => {
    it('seeds queued items and initial summary on start', () => {
        const { result } = renderUploadHook();

        act(() => {
            result.current.startUploadProgress([makeFile('a.jpg'), makeFile('b.jpg'), makeFile('c.jpg')]);
        });

        expect(result.current.uploadProgress).toEqual({
            total: 3,
            completed: 0,
            succeeded: 0,
            failed: 0,
            isUploading: true,
        });
    });

    it('shows a spinner toast immediately when upload starts', async () => {
        const { result } = renderUploadHook();

        act(() => {
            result.current.startUploadProgress([makeFile('a.jpg')]);
        });

        expect(await screen.findByText('Uploading 1 item...')).toBeVisible();
        expect(screen.getByLabelText('Loading...')).toBeVisible();
    });

    it('exposes a Show details action button on the in-progress toast', async () => {
        const { result } = renderUploadHook();

        act(() => {
            result.current.startUploadProgress([makeFile('a.jpg')]);
        });

        expect(await screen.findByRole('button', { name: 'Show details' })).toBeVisible();
    });

    it('removes the spinner when upload finishes', async () => {
        const { result } = renderUploadHook();

        act(() => {
            const [id] = result.current.startUploadProgress([makeFile('a.jpg')]);
            result.current.setItemUploading(id);
            result.current.setItemUploaded(id);
            result.current.finishUploadProgress();
        });

        expect(await screen.findByText('Uploaded 1 item')).toBeVisible();
        expect(screen.queryByLabelText('Loading...')).toBeNull();
    });

    it('shows only succeeded count when there are no failures yet', async () => {
        const { result } = renderUploadHook();

        act(() => {
            const ids = result.current.startUploadProgress([makeFile('a.jpg'), makeFile('b.jpg'), makeFile('c.jpg')]);
            result.current.setItemUploaded(ids[0]);
        });

        expect(await screen.findByText('Uploading 3 items... (1 succeeded)')).toBeVisible();
    });

    it('shows only failed count when there are no successes yet', async () => {
        const { result } = renderUploadHook();

        act(() => {
            const ids = result.current.startUploadProgress([makeFile('a.jpg'), makeFile('b.jpg'), makeFile('c.jpg')]);
            result.current.setItemFailed(ids[0], 'bad');
        });

        expect(await screen.findByText('Uploading 3 items... (1 failed)')).toBeVisible();
    });

    it('shows both counts when there are mixed results', async () => {
        const { result } = renderUploadHook();

        act(() => {
            const ids = result.current.startUploadProgress([makeFile('a.jpg'), makeFile('b.jpg'), makeFile('c.jpg')]);
            result.current.setItemUploaded(ids[0]);
            result.current.setItemFailed(ids[1], 'bad');
        });

        expect(await screen.findByText('Uploading 3 items... (1 succeeded, 1 failed)')).toBeVisible();
    });

    it('finishes upload progress and shows success toast', async () => {
        const { result } = renderUploadHook();

        act(() => {
            const ids = result.current.startUploadProgress([makeFile('a.jpg'), makeFile('b.jpg')]);
            result.current.setItemUploaded(ids[0]);
            result.current.setItemUploaded(ids[1]);
            result.current.finishUploadProgress();
        });

        expect(result.current.uploadProgress).toEqual({
            total: 2,
            completed: 2,
            succeeded: 2,
            failed: 0,
            isUploading: false,
        });
        expect(await screen.findByText('Uploaded 2 items')).toBeVisible();
    });

    it('finishes upload progress and shows warning toast for mixed results', async () => {
        const { result } = renderUploadHook();

        act(() => {
            const ids = result.current.startUploadProgress([makeFile('a.jpg'), makeFile('b.jpg')]);
            result.current.setItemUploaded(ids[0]);
            result.current.setItemFailed(ids[1], 'bad');
            result.current.finishUploadProgress();
        });

        expect(await screen.findByText('Uploaded 1 item, 1 failed')).toBeVisible();
    });

    it('finishes upload progress and shows error toast when all fail', async () => {
        const { result } = renderUploadHook();

        act(() => {
            const ids = result.current.startUploadProgress([makeFile('a.jpg'), makeFile('b.jpg')]);
            result.current.setItemFailed(ids[0], 'bad-1');
            result.current.setItemFailed(ids[1], 'bad-2');
            result.current.finishUploadProgress();
        });

        expect(await screen.findByText('Failed to upload 2 items')).toBeVisible();
    });

    it('appends items across subsequent uploads (persists history)', () => {
        const { result } = renderUploadHook();

        act(() => {
            result.current.startUploadProgress([makeFile('a.jpg')]);
        });

        act(() => {
            result.current.finishUploadProgress();
        });

        act(() => {
            result.current.startUploadProgress([makeFile('b.jpg'), makeFile('c.jpg')]);
        });

        expect(result.current.uploadProgress.total).toBe(3);
    });
});
