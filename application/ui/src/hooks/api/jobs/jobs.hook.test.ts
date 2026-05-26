// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { renderHook } from 'test-utils/render';

import { getMockedJob } from '../../../../mocks/mock-job';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { MockEventSourceConstructor, resetMockEventSource } from '../../../test-utils/mock-event-source';
import { useGetCurrentRunningJobs } from './jobs.hook';

const PROJECT_ID = '123';

const createMockJobForProject = (overrides: Partial<ReturnType<typeof getMockedJob>> = {}) =>
    getMockedJob({
        metadata: {
            project: { id: PROJECT_ID },
            model: {
                id: 'ef3983f1-cef0-4ebe-91db-7330f1dd6e27',
                name: 'SSD (ef3983f1)',
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

describe('useGetCurrentRunningJobs', () => {
    beforeEach(() => {
        resetMockEventSource();
    });

    it('returns an empty array when there are no active running jobs', async () => {
        server.use(http.get('/api/jobs', () => HttpResponse.json([])));

        const { result } = renderHook(() => useGetCurrentRunningJobs());

        await waitFor(() => {
            expect(result.current).toEqual([]);
        });
    });

    it('returns the active running job for the current project', async () => {
        const job = createMockJobForProject();
        server.use(http.get('/api/jobs', () => HttpResponse.json([job])));

        const { result } = renderHook(() => useGetCurrentRunningJobs());

        await waitFor(() => {
            expect(result.current).toHaveLength(1);
            expect(result.current?.[0].job_id).toBe(job.job_id);
        });
    });

    it('ignores jobs from other projects', async () => {
        const otherProjectJob = getMockedJob({
            metadata: {
                project: { id: 'other-project' },
                model: {
                    id: 'ef3983f1-cef0-4ebe-91db-7330f1dd6e27',
                    name: 'SSD (ef3983f1)',
                    architecture: 'SSD',
                    dataset_revision_id: 'ds-rev-1',
                },
                device: {
                    type: 'cpu',
                    name: 'CPU',
                },
            },
        });
        server.use(http.get('/api/jobs', () => HttpResponse.json([otherProjectJob])));

        const { result } = renderHook(() => useGetCurrentRunningJobs());

        await waitFor(() => {
            expect(result.current).toEqual([]);
        });
    });

    it('does not subscribe to SSE when no active running job matches the project', async () => {
        server.use(http.get('/api/jobs', () => HttpResponse.json([])));

        renderHook(() => useGetCurrentRunningJobs());

        await waitFor(() => {
            expect(MockEventSourceConstructor).not.toHaveBeenCalled();
        });
    });
});
