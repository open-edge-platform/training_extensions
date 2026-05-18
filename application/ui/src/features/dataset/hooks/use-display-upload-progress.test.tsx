// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, screen } from '@testing-library/react';
import { renderHook } from 'test-utils/render';

import { useUploadProgress } from './use-display-upload-progress';

const fulfilledResult = (value: unknown): PromiseFulfilledResult<unknown> => ({
    status: 'fulfilled',
    value,
});

const rejectedResult = (reason: unknown): PromiseRejectedResult => ({
    status: 'rejected',
    reason,
});

describe('useUploadProgress', () => {
    it('starts upload progress', () => {
        const { result } = renderHook(() => useUploadProgress());

        act(() => {
            result.current.startUploadProgress(3);
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
        const { result } = renderHook(() => useUploadProgress());

        act(() => {
            result.current.startUploadProgress(1);
        });

        expect(await screen.findByText('Uploading 1 item(s)...')).toBeVisible();
        expect(screen.getByLabelText('Loading...')).toBeVisible();
    });

    it('removes the spinner when upload finishes', async () => {
        const { result } = renderHook(() => useUploadProgress());

        act(() => {
            result.current.startUploadProgress(1);
            result.current.updateUploadProgress({ settledResults: [{ status: 'fulfilled', value: 'a' }] });
            result.current.finishUploadProgress();
        });

        expect(await screen.findByText('Uploaded 1 item(s)')).toBeVisible();
        expect(screen.queryByLabelText('Loading...')).toBeNull();
    });

    it('updates progress from settled batch results', () => {
        const { result } = renderHook(() => useUploadProgress());

        act(() => {
            result.current.startUploadProgress(3);
            result.current.updateUploadProgress({
                settledResults: [fulfilledResult('a'), rejectedResult(new Error('bad'))],
            });
        });

        expect(result.current.uploadProgress).toEqual({
            total: 3,
            completed: 2,
            succeeded: 1,
            failed: 1,
            isUploading: true,
        });
    });

    it('shows only succeeded count when there are no failures yet', async () => {
        const { result } = renderHook(() => useUploadProgress());

        act(() => {
            result.current.startUploadProgress(3);
            result.current.updateUploadProgress({ settledResults: [fulfilledResult('a')] });
        });

        expect(await screen.findByText('Uploading 3 item(s)... (1 succeeded)')).toBeVisible();
    });

    it('shows only failed count when there are no successes yet', async () => {
        const { result } = renderHook(() => useUploadProgress());

        act(() => {
            result.current.startUploadProgress(3);
            result.current.updateUploadProgress({ settledResults: [rejectedResult(new Error('bad'))] });
        });

        expect(await screen.findByText('Uploading 3 item(s)... (1 failed)')).toBeVisible();
    });

    it('shows both counts when there are mixed results', async () => {
        const { result } = renderHook(() => useUploadProgress());

        act(() => {
            result.current.startUploadProgress(3);
            result.current.updateUploadProgress({
                settledResults: [fulfilledResult('a'), rejectedResult(new Error('bad'))],
            });
        });

        expect(await screen.findByText('Uploading 3 item(s)... (1 succeeded, 1 failed)')).toBeVisible();
    });

    it('marks progress as finished on final batch update', () => {
        const { result } = renderHook(() => useUploadProgress());

        act(() => {
            result.current.startUploadProgress(2);
            result.current.updateUploadProgress({
                settledResults: [fulfilledResult('a')],
            });
            result.current.updateUploadProgress({
                settledResults: [fulfilledResult('b')],
            });
            result.current.finishUploadProgress();
        });

        expect(result.current.uploadProgress.isUploading).toBe(false);
        expect(result.current.uploadProgress.completed).toBe(2);
    });

    it('finishes upload progress and shows success toast', async () => {
        const { result } = renderHook(() => useUploadProgress());

        act(() => {
            result.current.startUploadProgress(2);
            result.current.updateUploadProgress({
                settledResults: [fulfilledResult('a'), fulfilledResult('b')],
            });
            result.current.finishUploadProgress();
        });

        expect(result.current.uploadProgress).toEqual({
            total: 2,
            completed: 2,
            succeeded: 2,
            failed: 0,
            isUploading: false,
        });
        expect(await screen.findByText('Uploaded 2 item(s)')).toBeVisible();
    });

    it('finishes upload progress and shows warning toast for mixed results', async () => {
        const { result } = renderHook(() => useUploadProgress());

        act(() => {
            result.current.startUploadProgress(2);
            result.current.updateUploadProgress({
                settledResults: [fulfilledResult('a'), rejectedResult(new Error('bad'))],
            });
            result.current.finishUploadProgress();
        });

        expect(result.current.uploadProgress).toEqual({
            total: 2,
            completed: 2,
            succeeded: 1,
            failed: 1,
            isUploading: false,
        });
        expect(await screen.findByText('Uploaded 1 item(s), 1 failed')).toBeVisible();
    });

    it('finishes upload progress and shows error toast when all fail', async () => {
        const { result } = renderHook(() => useUploadProgress());

        act(() => {
            result.current.startUploadProgress(2);
            result.current.updateUploadProgress({
                settledResults: [rejectedResult(new Error('bad-1')), rejectedResult(new Error('bad-2'))],
            });
            result.current.finishUploadProgress();
        });

        expect(result.current.uploadProgress).toEqual({
            total: 2,
            completed: 2,
            succeeded: 0,
            failed: 2,
            isUploading: false,
        });
        expect(await screen.findByText('Failed to upload 2 item(s)')).toBeVisible();
    });
});
