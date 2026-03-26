// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { renderHook } from 'test-utils/render';

import { getMockedJob } from '../../../../mocks/mock-job';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import {
    getLastEventSource,
    MockEventSourceConstructor,
    resetMockEventSource,
    simulateSSEMessage,
} from '../../../test-utils/mock-event-source';
import { useGetCurrentRunningJob } from './jobs.hook';

const PROJECT_ID = '123';

const createMockJobForProject = (overrides: Partial<ReturnType<typeof getMockedJob>> = {}) =>
    getMockedJob({
        metadata: {
            project: { id: PROJECT_ID },
            model: {
                id: 'model-1',
                architecture: 'SSD',
                dataset_revision_id: 'ds-rev-1',
            },
            device: {
                type: 'cpu',
                name: 'CPU',
            },
        },
        ...overrides,
    });

describe('useStreamJobStatus', () => {
    beforeEach(() => {
        resetMockEventSource();
    });

    it('updates the React Query cache when an SSE message arrives', async () => {
        const initialJob = createMockJobForProject({ progress: 10 });
        server.use(http.get('/api/jobs', () => HttpResponse.json([initialJob])));

        const { result: jobsResult } = renderHook(() => useGetCurrentRunningJob());

        await waitFor(() => {
            expect(jobsResult.current).toBeDefined();
        });

        const es = getLastEventSource();
        const updatedJob = createMockJobForProject({ progress: 50, message: 'Epoch 5/10' });

        act(() => {
            simulateSSEMessage(es, updatedJob);
        });

        await waitFor(() => {
            expect(jobsResult.current?.progress).toBe(50);
            expect(jobsResult.current?.message).toBe('Epoch 5/10');
        });
    });

    it('closes the connection when a terminal status is received', async () => {
        const initialJob = createMockJobForProject();
        server.use(http.get('/api/jobs', () => HttpResponse.json([initialJob])));

        renderHook(() => useGetCurrentRunningJob());

        await waitFor(() => {
            expect(MockEventSourceConstructor).toHaveBeenCalled();
        });

        const es = getLastEventSource();
        const completedJob = createMockJobForProject({ status: 'DONE', progress: 100 });

        act(() => {
            simulateSSEMessage(es, completedJob);
        });

        expect(es.close).toHaveBeenCalled();
    });
});

describe('useGetCurrentRunningJob', () => {
    beforeEach(() => {
        resetMockEventSource();
    });

    it('returns undefined when there are no active training jobs', async () => {
        server.use(http.get('/api/jobs', () => HttpResponse.json([])));

        const { result } = renderHook(() => useGetCurrentRunningJob());

        await waitFor(() => {
            expect(result.current).toBeUndefined();
        });
    });

    it('returns the active training job for the current project', async () => {
        const job = createMockJobForProject();
        server.use(http.get('/api/jobs', () => HttpResponse.json([job])));

        const { result } = renderHook(() => useGetCurrentRunningJob());

        await waitFor(() => {
            expect(result.current?.job_id).toBe(job.job_id);
        });
    });

    it('ignores jobs from other projects', async () => {
        const otherProjectJob = getMockedJob({
            metadata: {
                project: { id: 'other-project' },
                model: { id: 'model-1', architecture: 'SSD', dataset_revision_id: 'ds-rev-1' },
                device: {
                    type: 'cpu',
                    name: 'CPU',
                },
            },
        });
        server.use(http.get('/api/jobs', () => HttpResponse.json([otherProjectJob])));

        const { result } = renderHook(() => useGetCurrentRunningJob());

        await waitFor(() => {
            expect(result.current).toBeUndefined();
        });
    });

    it('subscribes to SSE when an active training job is found', async () => {
        const job = createMockJobForProject();
        server.use(http.get('/api/jobs', () => HttpResponse.json([job])));

        renderHook(() => useGetCurrentRunningJob());

        await waitFor(() => {
            expect(MockEventSourceConstructor).toHaveBeenCalled();
            expect(getLastEventSource().url).toContain(`/api/jobs/${job.job_id}/status`);
        });
    });

    it('does not subscribe to SSE when no active training job matches the project', async () => {
        server.use(http.get('/api/jobs', () => HttpResponse.json([])));

        renderHook(() => useGetCurrentRunningJob());

        await waitFor(() => {
            expect(MockEventSourceConstructor).not.toHaveBeenCalled();
        });
    });
});
