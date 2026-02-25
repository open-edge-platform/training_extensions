// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { queryClient } from '../../../query-client/query-client';
import { TogglePipelineButton } from './toggle-pipeline-button.component';

describe('TogglePipelineButton', () => {
    beforeEach(() => {
        queryClient.removeQueries();
    });

    it('enables pipeline when currently idle', async () => {
        let currentStatus: 'idle' | 'running' = 'idle';

        server.use(
            http.get('/api/projects/{project_id}/pipeline', () => {
                return HttpResponse.json(getMockedPipeline({ status: currentStatus }));
            }),
            http.post('/api/projects/{project_id}/pipeline:enable', () => {
                currentStatus = 'running';
                return HttpResponse.json(null, { status: 204 });
            }),
            http.post('/api/projects/{project_id}/pipeline:disable', () => {
                currentStatus = 'idle';
                return HttpResponse.json(null, { status: 204 });
            })
        );

        render(<TogglePipelineButton />);

        fireEvent.click(await screen.findByRole('button', { name: 'Enable Pipeline' }));

        expect(await screen.findByText('Pipeline enabled successfully')).toBeVisible();
    });

    it('disables pipeline when currently running', async () => {
        let currentStatus: 'idle' | 'running' = 'running';

        server.use(
            http.get('/api/projects/{project_id}/pipeline', () => {
                return HttpResponse.json(getMockedPipeline({ status: currentStatus }));
            }),
            http.post('/api/projects/{project_id}/pipeline:enable', () => {
                currentStatus = 'running';
                return HttpResponse.json(null, { status: 204 });
            }),
            http.post('/api/projects/{project_id}/pipeline:disable', () => {
                currentStatus = 'idle';
                return HttpResponse.json(null, { status: 204 });
            })
        );

        render(<TogglePipelineButton />);

        fireEvent.click(await screen.findByRole('button', { name: 'Disable Pipeline' }));

        expect(await screen.findByText('Pipeline disabled successfully')).toBeVisible();
    });

    it('shows explanation dialog when pipeline is not configured', async () => {
        server.use(
            http.get('/api/projects/{project_id}/pipeline', () => {
                return HttpResponse.json(
                    getMockedPipeline({
                        status: 'idle',
                        model: null,
                        source: null,
                        sink: null,
                    })
                );
            })
        );

        render(<TogglePipelineButton />);

        fireEvent.click(await screen.findByRole('button', { name: 'Enable Pipeline' }));

        expect(await screen.findByText('Cannot enable pipeline')).toBeVisible();
        expect(
            await screen.findByText('Make sure you selected a model, source, and sink before enabling the pipeline.')
        ).toBeVisible();
    });
});
