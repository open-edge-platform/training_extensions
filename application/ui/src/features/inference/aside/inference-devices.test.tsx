// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { InferenceDevices } from './inference-devices.component';

const mockPipeline = getMockedPipeline({
    device: 'cpu',
});

const mockDevices = [
    { type: 'cpu' as const, name: 'CPU' },
    { type: 'xpu' as const, name: 'GPU' },
];

describe('InferenceDevices', () => {
    const renderApp = (status = 200, devicesResponse = mockDevices, pipelineResponse = mockPipeline) => {
        const pipelinePatchSpy = vi.fn();

        server.use(
            http.get('/api/system/devices/inference', () => HttpResponse.json(devicesResponse)),
            http.get('/api/projects/{project_id}/pipeline', () => HttpResponse.json(pipelineResponse)),
            http.patch('/api/projects/{project_id}/pipeline', () => {
                pipelinePatchSpy();
                return HttpResponse.json(
                    {
                        project_id: '',
                        status: 'idle',
                        device: 'images_folder',
                    },
                    { status }
                );
            })
        );

        render(<InferenceDevices />);

        return pipelinePatchSpy;
    };

    beforeEach(() => {
        vi.resetAllMocks();
    });

    it('displays current device selection', async () => {
        renderApp(200, mockDevices, { ...mockPipeline, device: 'xpu' });

        expect(await screen.findByLabelText('inference compute')).toHaveTextContent('GPU');
    });

    it('updates device on selection change', async () => {
        const pipelinePatchSpy = renderApp();

        const picker = await screen.findByLabelText('inference compute');
        expect(picker).toHaveTextContent('CPU');

        const button = screen.getByRole('button', { name: /inference compute/i });
        await userEvent.click(button);

        const option = await screen.findByRole('option', { name: /GPU/i });
        await userEvent.click(option);

        await waitFor(() => {
            expect(screen.getByLabelText('inference compute')).toHaveTextContent('GPU');
            expect(pipelinePatchSpy).toHaveBeenCalled();
        });
    });

    it('shows error toast on update failure', async () => {
        renderApp(500);

        await screen.findByLabelText('inference compute');

        await userEvent.click(screen.getByRole('button', { name: /inference compute/i }));
        await userEvent.click(screen.getByRole('option', { name: /GPU/i }));

        expect(await screen.findByLabelText('toast')).toBeVisible();
    });

    it('reverts selection on error', async () => {
        renderApp(500, mockDevices, { ...mockPipeline, device: 'cpu' });

        expect(await screen.findByLabelText('inference compute')).toHaveTextContent('CPU');

        await userEvent.click(screen.getByRole('button', { name: /inference compute/i }));
        await userEvent.click(screen.getByRole('option', { name: /GPU/i }));

        await screen.findByLabelText('toast');

        expect(await screen.findByLabelText('inference compute')).toHaveTextContent('CPU');
    });
});
