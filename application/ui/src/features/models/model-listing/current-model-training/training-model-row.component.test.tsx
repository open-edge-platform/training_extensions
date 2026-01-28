// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { getMockedModel } from 'mocks/mock-model';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { getMockedJob } from '../../../../../mocks/mock-job';
import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { TrainingModelRow } from './training-model-row.component';

describe('TrainingModelRow', () => {
    const mockModels = [
        getMockedModel({
            id: 'model-123',
            name: 'My Detection Model',
        }),
    ];

    beforeEach(() => {
        server.use(
            http.get('/api/projects/{project_id}/models', () => {
                return HttpResponse.json(mockModels);
            })
        );
    });

    it('renders all fields correctly', async () => {
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

        render(<TrainingModelRow job={job} />);

        expect(await screen.findByText('My Detection Model')).toBeVisible();
        expect(screen.getByText('Custom_Object_Detection_Gen3_ATSS')).toBeVisible();
        expect(screen.getByText('Training')).toBeVisible();
        expect(screen.getByText('Running...')).toBeVisible();
        expect(screen.getByText(/Started: 19 Jan 2026/i)).toBeVisible();
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

        render(<TrainingModelRow job={job} onCancel={mockCancel} />);

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

        render(<TrainingModelRow job={job} onCancel={mockCancel} />);

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

        render(<TrainingModelRow job={job} />);

        expect(await screen.findByText('unknown-model-id')).toBeVisible();
    });
});
