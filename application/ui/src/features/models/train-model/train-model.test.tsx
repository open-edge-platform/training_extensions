// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { TrainModel } from './train-model.component';

describe('TrainModel', () => {
    it('disables train model button when there are no enough annotated media items', async () => {
        server.use(
            http.get('/api/projects/{project_id}/dataset/items', () => {
                return HttpResponse.json({
                    items: [
                        {
                            id: '1',
                            subset: 'unassigned',
                        },
                        {
                            id: '2',
                            subset: 'unassigned',
                        },
                        {
                            id: '3',
                            subset: 'unassigned',
                        },
                    ],
                    pagination: {
                        total: 3,
                        count: 3,
                        limit: 10,
                        offset: 0,
                    },
                });
            })
        );

        render(<TrainModel />);

        await waitFor(() => {
            expect(screen.getByRole('button', { name: 'Train model' })).toBeDisabled();
        });
    });

    it('enables train model button when there are enough annotated media items', async () => {
        server.use(
            http.get('/api/projects/{project_id}/dataset/items', () => {
                return HttpResponse.json({
                    items: [
                        {
                            id: '1',
                            subset: 'unassigned',
                        },
                        {
                            id: '2',
                            subset: 'unassigned',
                        },
                        {
                            id: '3',
                            subset: 'unassigned',
                        },
                        {
                            id: '4',
                            subset: 'unassigned',
                        },
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

        await waitFor(() => {
            expect(screen.getByRole('button', { name: 'Train model' })).toBeEnabled();
        });
    });
});
