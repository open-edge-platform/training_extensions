// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, screen, waitFor, within } from '@testing-library/react';
import { getMockedDatasetRevision } from 'mocks/mock-dataset-revision';
import { getMockedJob } from 'mocks/mock-job';
import { getMockedModel, getMockedModelArchitecture } from 'mocks/mock-model';
import { HttpResponse } from 'msw';
import { render, renderHook } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { useGetCurrentRunningJobs, useStreamJobStatus } from '../../../../hooks/api/jobs/jobs.hook';
import { server } from '../../../../msw-node-setup';
import {
    getLastEventSource,
    MockEventSourceConstructor,
    resetMockEventSource,
    simulateSSEMessage,
} from '../../../../test-utils/mock-event-source';
import { RunningModelRow } from './running-model-row.component';

describe('RunningModelRow', () => {
    const mockModel = getMockedModel({
        id: 'model-123',
        architecture: 'arch-123',
        name: 'My Detection Model',
        training_info: {
            dataset_revision_id: 'dataset-123',
            status: 'in_progress',
            label_schema_revision: {
                labels: [
                    { id: '1', name: 'car' },
                    { id: '2', name: 'person' },
                ],
            },
        },
    });

    const modelArchitecture = getMockedModelArchitecture({
        performanceCategory: 'Speed',
        id: mockModel.architecture,
        name: 'Custom_Object_Detection_Gen3_ATSS',
    });

    beforeEach(() => {
        resetMockEventSource();
        server.use(
            http.get('/api/projects/{project_id}/models/{model_id}', () => {
                return HttpResponse.json(mockModel);
            })
        );
    });

    it('renders all fields correctly when grouped by dataset', async () => {
        const job = getMockedJob({
            metadata: {
                project: { id: '123' },
                model: {
                    id: mockModel.id,
                    name: mockModel.name,
                    architecture: modelArchitecture.id,
                    parent_revision_id: null,
                    dataset_revision_id: 'dataset-123',
                },
                device: {
                    type: 'cpu',
                    name: 'CPU',
                },
            },
            status: 'RUNNING',
            message: 'Running...',
            started_at: '2026-01-19T08:15:00.000000+00:00',
        });

        const datasetRevision = getMockedDatasetRevision({
            id: 'dataset-123',
            name: 'Dataset 1',
            item_counts: {
                total: 10,
                testing: 4,
                training: 4,
                validation: 2,
            },
        });

        render(
            <RunningModelRow
                job={job}
                groupBy={'dataset'}
                datasetRevisions={[datasetRevision]}
                modelArchitectures={[modelArchitecture]}
            />
        );

        expect(await screen.findByText('My Detection Model')).toBeVisible();
        expect(screen.getByText('Running...')).toBeVisible();
        expect(screen.getByText(/Started: 19 Jan 2026/i)).toBeVisible();
        expect(screen.getByText('Device: CPU')).toBeVisible();

        expect(screen.getByText(new RegExp(modelArchitecture.name))).toBeVisible();
        expect(screen.queryByText(datasetRevision.name)).not.toBeInTheDocument();
        expect(screen.queryByTestId('dataset-count')).not.toBeInTheDocument();
        expect(screen.queryByTestId('labels-count')).not.toBeInTheDocument();
    });

    it('renders all fields correctly when grouped by architecture', async () => {
        const job = getMockedJob({
            metadata: {
                project: { id: '123' },
                model: {
                    id: 'ef3983f1-cef0-4ebe-91db-7330f1dd6e27',
                    name: 'ATSS (ef3983f1)',
                    architecture: 'Custom_Object_Detection_Gen3_ATSS',
                    parent_revision_id: null,
                    dataset_revision_id: 'dataset-123',
                },
                device: {
                    type: 'cpu',
                    name: 'CPU',
                },
            },
            status: 'RUNNING',
            message: 'Running...',
            started_at: '2026-01-19T08:15:00.000000+00:00',
        });

        const datasetRevision = getMockedDatasetRevision({
            id: 'dataset-123',
            name: 'Dataset 1',
            item_counts: {
                total: 10,
                testing: 4,
                training: 4,
                validation: 2,
            },
        });

        render(
            <RunningModelRow
                job={job}
                groupBy={'architecture'}
                datasetRevisions={[datasetRevision]}
                modelArchitectures={[modelArchitecture]}
            />
        );

        expect(await screen.findByText('My Detection Model')).toBeVisible();
        expect(screen.getByText('Running')).toBeVisible();
        expect(screen.getByText('Running...')).toBeVisible();
        expect(screen.getByText(/Started: 19 Jan 2026/i)).toBeVisible();
        expect(screen.getByText('Device: CPU')).toBeVisible();

        expect(screen.queryByText(new RegExp(modelArchitecture.name))).not.toBeInTheDocument();

        expect(screen.getByText(datasetRevision.name)).toBeInTheDocument();
        const datasetBadge = screen.getByTestId('dataset-count');
        expect(within(datasetBadge).getByText(datasetRevision.item_counts?.total?.toString() ?? ''));

        const labelsBadge = screen.getByTestId('labels-count');
        const labelSchemaRevision = mockModel.training_info.label_schema_revision ?? {};
        const labelsCount =
            'labels' in labelSchemaRevision && Array.isArray(labelSchemaRevision.labels)
                ? labelSchemaRevision.labels.length
                : '';
        expect(within(labelsBadge).getByText(labelsCount));
    });

    it('renders Cancel button when onCancel is provided and job is running', async () => {
        const mockCancel = vi.fn();
        const job = getMockedJob({
            metadata: {
                project: { id: '123' },
                model: {
                    id: 'ef3983f1-cef0-4ebe-91db-7330f1dd6e27',
                    name: 'ATSS (ef3983f1)',
                    architecture: 'Custom_Object_Detection_Gen3_ATSS',
                    parent_revision_id: null,
                    dataset_revision_id: 'dataset-123',
                },
                device: {
                    type: 'cpu',
                    name: 'CPU',
                },
            },
            status: 'RUNNING',
        });

        render(
            <RunningModelRow
                job={job}
                onCancel={mockCancel}
                datasetRevisions={[]}
                groupBy={'dataset'}
                modelArchitectures={[modelArchitecture]}
            />
        );

        const cancelButton = await screen.findByRole('button', { name: /cancel job/i });
        expect(cancelButton).toBeVisible();
        expect(cancelButton).toBeEnabled();
    });

    it('disables Cancel button when job is not running', async () => {
        const mockCancel = vi.fn();
        const job = getMockedJob({
            metadata: {
                project: { id: '123' },
                model: {
                    id: 'ef3983f1-cef0-4ebe-91db-7330f1dd6e27',
                    name: 'ATSS (ef3983f1)',
                    architecture: 'Custom_Object_Detection_Gen3_ATSS',
                    parent_revision_id: null,
                    dataset_revision_id: 'dataset-123',
                },
                device: {
                    type: 'cpu',
                    name: 'CPU',
                },
            },
            status: 'FINISHED',
        });

        render(
            <RunningModelRow
                job={job}
                onCancel={mockCancel}
                datasetRevisions={[]}
                groupBy={'dataset'}
                modelArchitectures={[modelArchitecture]}
            />
        );

        const cancelButton = await screen.findByRole('button', { name: /cancel job/i });
        expect(cancelButton).toBeDisabled();
    });

    describe('useStreamJobStatus', () => {
        const job = getMockedJob({
            metadata: {
                project: { id: '123' },
                model: {
                    id: 'ef3983f1-cef0-4ebe-91db-7330f1dd6e27',
                    name: 'ATSS (ef3983f1)',
                    architecture: 'Custom_Object_Detection_Gen3_ATSS',
                    parent_revision_id: null,
                    dataset_revision_id: 'dataset-123',
                },
                device: {
                    type: 'cpu',
                    name: 'CPU',
                },
            },
            status: 'RUNNING',
        });

        beforeEach(() => {
            resetMockEventSource();
        });

        it('subscribes to SSE when the component mounts with a running job', async () => {
            render(
                <RunningModelRow
                    job={job}
                    datasetRevisions={[]}
                    groupBy={'dataset'}
                    modelArchitectures={[modelArchitecture]}
                />
            );

            await waitFor(() => {
                expect(MockEventSourceConstructor).toHaveBeenCalled();
                expect(getLastEventSource().url).toContain(`/api/jobs/${job.job_id}/status`);
            });
        });

        it('updates the React Query cache when an SSE message arrives', async () => {
            server.use(http.get('/api/jobs', () => HttpResponse.json([job])));

            const { result: jobsResult } = renderHook(() => {
                useStreamJobStatus(job.job_id);
                return useGetCurrentRunningJobs();
            });

            await waitFor(() => {
                expect(jobsResult.current).toBeDefined();
            });

            const es = getLastEventSource();
            const updatedJob = { ...job, progress: 50, message: 'Epoch 5/10' };

            act(() => {
                simulateSSEMessage(es, updatedJob);
            });

            await waitFor(() => {
                expect(jobsResult.current?.[0].progress).toBe(50);
                expect(jobsResult.current?.[0].message).toBe('Epoch 5/10');
            });
        });

        it('closes the SSE connection when a terminal status is received', async () => {
            server.use(http.get('/api/jobs', () => HttpResponse.json([job])));

            renderHook(() => {
                useStreamJobStatus(job.job_id);
            });

            await waitFor(() => {
                expect(MockEventSourceConstructor).toHaveBeenCalled();
            });

            const es = getLastEventSource();
            const completedJob = { ...job, status: 'DONE' as const, progress: 100 };

            act(() => {
                simulateSSEMessage(es, completedJob);
            });

            expect(es.close).toHaveBeenCalled();
        });
    });
});
