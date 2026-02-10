// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { renderHook } from 'test-utils/render';

import { http } from '../../api/utils';
import type { Job } from '../../constants/shared-types';
import { server } from '../../msw-node-setup';
import { queryClient } from '../../query-client/query-client';
import { useGetCurrentTrainingJob, useStreamJobStatus } from './jobs.hook';

type MockEventSource = {
    onopen: ((event: Event) => void) | null;
    onmessage: ((event: MessageEvent) => void) | null;
    onerror: ((event: Event) => void) | null;
    close: ReturnType<typeof vi.fn>;
    url: string;
};

let mockEventSourceInstances: MockEventSource[] = [];

const MockEventSourceConstructor = vi.fn().mockImplementation((url: string) => {
    const instance: MockEventSource = {
        onopen: null,
        onmessage: null,
        onerror: null,
        close: vi.fn(),
        url,
    };
    mockEventSourceInstances.push(instance);
    return instance;
});

vi.stubGlobal('EventSource', MockEventSourceConstructor);

const getLastEventSource = (): MockEventSource => {
    const instance = mockEventSourceInstances.at(-1);
    if (!instance) {
        throw new Error('No MockEventSource instance found');
    }
    return instance;
};

const simulateSSEMessage = (instance: MockEventSource, data: unknown) => {
    instance.onmessage?.({ data: JSON.stringify(data) } as MessageEvent);
};

const PROJECT_ID = '123';

const createMockJob = (overrides: Partial<Job> = {}): Job => ({
    job_id: 'job-1',
    job_type: 'train',
    status: 'RUNNING',
    progress: 0,
    message: 'Training...',
    error: null,
    started_at: '2026-02-06T10:00:00Z',
    finished_at: null,
    metadata: {
        project: { id: PROJECT_ID },
        model: {
            id: 'model-1',
            architecture: 'SSD',
            dataset_revision_id: 'ds-rev-1',
        },
    },
    ...overrides,
});

describe('useStreamJobStatus', () => {
    beforeEach(() => {
        mockEventSourceInstances = [];
        MockEventSourceConstructor.mockClear();
        queryClient.clear();
    });

    it('does not create an EventSource when jobId is undefined', () => {
        renderHook(() => useStreamJobStatus(undefined));

        expect(MockEventSourceConstructor).not.toHaveBeenCalled();
    });

    it('creates an EventSource when jobId is provided', () => {
        renderHook(() => useStreamJobStatus('job-1'));

        expect(MockEventSourceConstructor).toHaveBeenCalledTimes(1);
        expect(getLastEventSource().url).toContain('/api/jobs/job-1/status');
    });

    it('closes the EventSource on unmount', () => {
        const { unmount } = renderHook(() => useStreamJobStatus('job-1'));
        const es = getLastEventSource();

        unmount();

        expect(es.close).toHaveBeenCalled();
    });

    it('updates the React Query cache when an SSE message arrives', async () => {
        const initialJob = createMockJob({ progress: 10 });
        server.use(http.get('/api/jobs', () => HttpResponse.json([initialJob])));

        const { result: jobsResult } = renderHook(() => useGetCurrentTrainingJob());

        await waitFor(() => {
            expect(jobsResult.current).toBeDefined();
        });

        // Now simulate an SSE update with higher progress
        const es = getLastEventSource();
        const updatedJob = createMockJob({ progress: 50, message: 'Epoch 5/10' });

        act(() => {
            simulateSSEMessage(es, updatedJob);
        });

        await waitFor(() => {
            expect(jobsResult.current?.progress).toBe(50);
            expect(jobsResult.current?.message).toBe('Epoch 5/10');
        });
    });

    it('closes the connection when a terminal status is received', async () => {
        const initialJob = createMockJob();
        server.use(http.get('/api/jobs', () => HttpResponse.json([initialJob])));

        renderHook(() => useGetCurrentTrainingJob());

        await waitFor(() => {
            expect(MockEventSourceConstructor).toHaveBeenCalled();
        });

        const es = getLastEventSource();
        const completedJob = createMockJob({ status: 'DONE', progress: 100 });

        act(() => {
            simulateSSEMessage(es, completedJob);
        });

        expect(es.close).toHaveBeenCalled();
    });
});

describe('useGetCurrentTrainingJob', () => {
    beforeEach(() => {
        mockEventSourceInstances = [];
        MockEventSourceConstructor.mockClear();
        queryClient.clear();
    });

    it('returns undefined when there are no active training jobs', async () => {
        server.use(http.get('/api/jobs', () => HttpResponse.json([])));

        const { result } = renderHook(() => useGetCurrentTrainingJob());

        await waitFor(() => {
            expect(result.current).toBeUndefined();
        });
    });

    it('returns the active training job for the current project', async () => {
        const job = createMockJob();
        server.use(http.get('/api/jobs', () => HttpResponse.json([job])));

        const { result } = renderHook(() => useGetCurrentTrainingJob());

        await waitFor(() => {
            expect(result.current?.job_id).toBe('job-1');
        });
    });

    it('ignores jobs from other projects', async () => {
        const otherProjectJob = createMockJob({
            metadata: {
                project: { id: 'other-project' },
                model: { id: 'model-1', architecture: 'SSD', dataset_revision_id: 'ds-rev-1' },
            },
        });
        server.use(http.get('/api/jobs', () => HttpResponse.json([otherProjectJob])));

        const { result } = renderHook(() => useGetCurrentTrainingJob());

        await waitFor(() => {
            expect(result.current).toBeUndefined();
        });
    });

    it('subscribes to SSE when an active training job is found', async () => {
        const job = createMockJob();
        server.use(http.get('/api/jobs', () => HttpResponse.json([job])));

        renderHook(() => useGetCurrentTrainingJob());

        await waitFor(() => {
            expect(MockEventSourceConstructor).toHaveBeenCalled();
            expect(getLastEventSource().url).toContain('/api/jobs/job-1/status');
        });
    });

    it('does not subscribe to SSE when no active training job matches the project', async () => {
        server.use(http.get('/api/jobs', () => HttpResponse.json([])));

        renderHook(() => useGetCurrentTrainingJob());

        await waitFor(() => {
            expect(mockEventSourceInstances).toHaveLength(0);
        });
    });
});
