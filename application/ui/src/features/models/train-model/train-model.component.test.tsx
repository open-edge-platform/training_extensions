// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen, waitFor } from '@testing-library/react';
import { getMockedDatasetItem } from 'mocks/mock-dataset-item';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { getMockedProject } from 'mocks/mock-project';
import { getMockedTrainingConfiguration } from 'mocks/mock-training-configuration';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { TrainModel } from './train-model.component';

describe('TrainModel', () => {
    beforeEach(() => {
        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(getMockedProject({ id: '123' }));
            }),
            http.get('/api/projects/{project_id}/pipeline', () => {
                return HttpResponse.json(getMockedPipeline({}));
            }),
            http.get('/api/projects/{project_id}/dataset_revisions', () => {
                return HttpResponse.json([]);
            }),
            http.get('/api/projects/{project_id}/models', () => {
                return HttpResponse.json([]);
            }),
            http.get('/api/projects/{project_id}/models/{model_id}/training_configuration', () => {
                return HttpResponse.json({ parameters: getMockedTrainingConfiguration() });
            }),
            http.get('/api/projects/{project_id}/training_configuration', () => {
                return HttpResponse.json({ parameters: getMockedTrainingConfiguration() });
            }),
            http.get('/api/model_architectures', () => {
                return HttpResponse.json({
                    model_architectures: [],
                    top_picks: null,
                });
            }),
            http.get('/api/system/devices/training', () => {
                return HttpResponse.json([{ type: 'cpu', name: 'CPU' }]);
            })
        );
    });

    it.skip('shows warning message when there are not enough annotated media items', async () => {
        server.use(
            http.get('/api/projects/{project_id}/dataset/items', () => {
                return HttpResponse.json({
                    items: [
                        getMockedDatasetItem({
                            id: '1',
                            subset: 'unassigned',
                        }),
                        getMockedDatasetItem({
                            id: '2',
                            subset: 'unassigned',
                        }),
                    ],
                    pagination: {
                        total: 2,
                        count: 2,
                        limit: 10,
                        offset: 0,
                    },
                });
            })
        );

        render(<TrainModel />);

        fireEvent.click(screen.getByRole('button', { name: 'Train model' }));

        expect(
            await screen.findByText(/In order to train a model, you need to annotate at least 3 items/)
        ).toBeVisible();
    });

    it('does not show warning message when there are enough annotated media items', async () => {
        server.use(
            http.get('/api/projects/{project_id}/dataset/items', () => {
                return HttpResponse.json({
                    items: [
                        getMockedDatasetItem({
                            id: '1',
                            subset: 'unassigned',
                        }),
                        getMockedDatasetItem({
                            id: '2',
                            subset: 'unassigned',
                        }),
                        getMockedDatasetItem({
                            id: '3',
                            subset: 'unassigned',
                        }),
                        getMockedDatasetItem({
                            id: '4',
                            subset: 'unassigned',
                        }),
                    ],
                    pagination: {
                        total: 4,
                        count: 4,
                        limit: 10,
                        offset: 0,
                    },
                });
            })
        );

        render(<TrainModel />);

        fireEvent.click(screen.getByRole('button', { name: 'Train model' }));

        await waitFor(() => {
            expect(
                screen.queryByText(/In order to train a model, you need to annotate at least 3 items/)
            ).not.toBeInTheDocument();
        });
    });
});
