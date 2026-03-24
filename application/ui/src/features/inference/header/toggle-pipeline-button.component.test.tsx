// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { TogglePipelineButton } from './toggle-pipeline-button.component';

describe('TogglePipelineButton', () => {
    it('enables pipeline when currently idle', async () => {
        server.use(
            http.get('/api/projects/{project_id}/pipeline', () => {
                return HttpResponse.json(getMockedPipeline({ status: 'idle' }));
            }),
            http.post('/api/projects/{project_id}/pipeline:enable', () => {
                return HttpResponse.json(null, { status: 204 });
            }),
            http.post('/api/projects/{project_id}/pipeline:disable', () => {
                return HttpResponse.json(null, { status: 204 });
            })
        );

        render(<TogglePipelineButton />);

        fireEvent.click(await screen.findByRole('button', { name: /Enable Pipeline/i }));

        expect(await screen.findByText('Pipeline enabled successfully')).toBeVisible();
    });

    it('disables pipeline when currently running', async () => {
        server.use(
            http.get('/api/projects/{project_id}/pipeline', () => {
                return HttpResponse.json(getMockedPipeline({ status: 'running' }));
            }),
            http.post('/api/projects/{project_id}/pipeline:enable', () => {
                return HttpResponse.json(null, { status: 204 });
            }),
            http.post('/api/projects/{project_id}/pipeline:disable', () => {
                return HttpResponse.json(null, { status: 204 });
            })
        );

        render(<TogglePipelineButton />);

        fireEvent.click(await screen.findByRole('button', { name: /Disable Pipeline/i }));

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

        fireEvent.click(await screen.findByRole('button', { name: /Enable Pipeline/i }));

        expect(await screen.findByText('Cannot enable pipeline')).toBeVisible();
        expect(
            await screen.findByText('Make sure you selected a model, source, and sink before enabling the pipeline.')
        ).toBeVisible();
    });
});
