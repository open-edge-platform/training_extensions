// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { renderHook } from 'test-utils/render';

import { getMockedPrepareImportDatasetJob } from '../../../mocks/mock-job';
import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { useImportJobStatus } from './jobs/use-import-job-status.hook';

describe('useImportJobStatus', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('polls job status and calls onSuccess when job finishes successfully', async () => {
        const onError = vi.fn();
        const onSuccess = vi.fn();

        server.use(
            http.get('/api/jobs/{job_id}', ({ params }) => {
                return HttpResponse.json(getMockedPrepareImportDatasetJob({ job_id: params.job_id, status: 'DONE' }), {
                    status: 200,
                });
            })
        );

        const { result } = renderHook(() => useImportJobStatus({ jobId: 'job-123', onSuccess, onError }));

        await waitFor(() => {
            expect(result.current.data?.status).toBe('DONE');
        });

        expect(onSuccess).toHaveBeenCalled();
        expect(onError).not.toHaveBeenCalled();
    });

    it('calls onError when job fails', async () => {
        const onError = vi.fn();
        const onSuccess = vi.fn();

        server.use(
            http.get('/api/jobs/{job_id}', ({ params }) => {
                return HttpResponse.json(
                    getMockedPrepareImportDatasetJob({ job_id: params.job_id, status: 'FAILED' }),
                    { status: 200 }
                );
            })
        );

        const { result } = renderHook(() => useImportJobStatus({ jobId: 'job-456', onSuccess, onError }));

        await waitFor(() => {
            expect(result.current.data?.status).toBe('FAILED');
        });

        expect(onError).toHaveBeenCalled();
        expect(onSuccess).not.toHaveBeenCalled();
    });

    it('handles running status without calling callbacks', async () => {
        const onError = vi.fn();
        const onSuccess = vi.fn();

        server.use(
            http.get('/api/jobs/{job_id}', ({ params }) => {
                return HttpResponse.json(
                    getMockedPrepareImportDatasetJob({
                        job_id: params.job_id,
                        status: 'RUNNING',
                        progress: 50,
                    }),
                    { status: 200 }
                );
            })
        );

        const { result } = renderHook(() => useImportJobStatus({ jobId: 'job-789', onSuccess, onError }));

        await waitFor(() => {
            expect(result.current.data?.progress).toBe(50);
        });

        expect(onSuccess).not.toHaveBeenCalled();
        expect(onError).not.toHaveBeenCalled();
    });

    it('does not poll when jobId is not provided', () => {
        const onError = vi.fn();
        const onSuccess = vi.fn();

        const { result } = renderHook(() => useImportJobStatus({ jobId: undefined, onSuccess, onError }));

        expect(result.current.data).toBeUndefined();
        expect(onSuccess).not.toHaveBeenCalled();
        expect(onError).not.toHaveBeenCalled();
    });
});
