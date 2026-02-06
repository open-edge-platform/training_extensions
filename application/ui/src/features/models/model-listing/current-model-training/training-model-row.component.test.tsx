// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, within } from '@testing-library/react';
import { getMockedDatasetRevision } from 'mocks/mock-dataset-revision';
import { getMockedModel } from 'mocks/mock-model';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { getMockedJob } from '../../../../../mocks/mock-job';
import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { TrainingModelRow } from './training-model-row.component';

describe('TrainingModelRow', () => {
    const mockModels = getMockedModel({
        id: 'model-123',
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

    beforeEach(() => {
        server.use(
            http.get('/api/projects/{project_id}/models/{model_id}', () => {
                return HttpResponse.json(mockModels);
            })
        );
    });

    it('renders all fields correctly when grouped by dataset', async () => {
        const job = getMockedJob({
            metadata: {
                project: { id: '123' },
                model: {
                    id: 'model-123',
                    architecture: 'Custom_Object_Detection_Gen3_ATSS',
                    parent_revision_id: null,
                    dataset_revision_id: 'dataset-123',
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

        render(<TrainingModelRow job={job} groupBy={'dataset'} datasetRevisions={[datasetRevision]} />);

        expect(await screen.findByText('My Detection Model')).toBeVisible();
        expect(screen.getByText('Training')).toBeVisible();
        expect(screen.getByText('Running...')).toBeVisible();
        expect(screen.getByText(/Started: 19 Jan 2026/i)).toBeVisible();

        expect(screen.getByText(/Custom_Object_Detection_Gen3_ATSS/)).toBeVisible();
        expect(screen.queryByText(datasetRevision.name)).not.toBeInTheDocument();
        expect(screen.queryByTestId('dataset-count')).not.toBeInTheDocument();
        expect(screen.queryByTestId('labels-count')).not.toBeInTheDocument();
    });

    it('renders all fields correctly when grouped by architecture', async () => {
        const job = getMockedJob({
            metadata: {
                project: { id: '123' },
                model: {
                    id: 'model-123',
                    architecture: 'Custom_Object_Detection_Gen3_ATSS',
                    parent_revision_id: null,
                    dataset_revision_id: 'dataset-123',
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

        render(<TrainingModelRow job={job} groupBy={'architecture'} datasetRevisions={[datasetRevision]} />);

        expect(await screen.findByText('My Detection Model')).toBeVisible();
        expect(screen.getByText('Training')).toBeVisible();
        expect(screen.getByText('Running...')).toBeVisible();
        expect(screen.getByText(/Started: 19 Jan 2026/i)).toBeVisible();

        expect(screen.queryByText(/Custom_Object_Detection_Gen3_ATSS/)).not.toBeInTheDocument();

        expect(screen.getByText(datasetRevision.name)).toBeInTheDocument();
        const datasetBadge = screen.getByTestId('dataset-count');
        expect(within(datasetBadge).getByText(datasetRevision.item_counts?.total?.toString() ?? ''));

        const labelsBadge = screen.getByTestId('labels-count');
        const labelSchemaRevision = mockModels.training_info.label_schema_revision ?? {};
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
                    id: 'model-123',
                    architecture: 'Custom_Object_Detection_Gen3_ATSS',
                    parent_revision_id: null,
                    dataset_revision_id: 'dataset-123',
                },
            },
            status: 'RUNNING',
        });

        render(<TrainingModelRow job={job} onCancel={mockCancel} datasetRevisions={[]} groupBy={'dataset'} />);

        const cancelButton = await screen.findByRole('button', { name: /cancel training job/i });
        expect(cancelButton).toBeVisible();
        expect(cancelButton).toBeEnabled();
    });

    it('disables Cancel button when job is not running', async () => {
        const mockCancel = vi.fn();
        const job = getMockedJob({
            metadata: {
                project: { id: '123' },
                model: {
                    id: 'model-123',
                    architecture: 'Custom_Object_Detection_Gen3_ATSS',
                    parent_revision_id: null,
                    dataset_revision_id: 'dataset-123',
                },
            },
            status: 'FINISHED',
        });

        render(<TrainingModelRow job={job} onCancel={mockCancel} datasetRevisions={[]} groupBy={'dataset'} />);

        const cancelButton = await screen.findByRole('button', { name: /cancel training job/i });
        expect(cancelButton).toBeDisabled();
    });

    it('falls back to model ID when model name is not found', async () => {
        server.use(
            http.get('/api/projects/{project_id}/models', () => {
                return HttpResponse.json([]);
            })
        );

        const job = getMockedJob({
            metadata: {
                project: { id: '123' },
                model: {
                    id: 'unknown-model-id',
                    architecture: 'Custom_Object_Detection_Gen3_ATSS',
                    parent_revision_id: null,
                    dataset_revision_id: 'dataset-123',
                },
            },
        });

        render(<TrainingModelRow job={job} datasetRevisions={[]} groupBy={'dataset'} />);

        expect(await screen.findByText('unknown-model-id')).toBeVisible();
    });
});
