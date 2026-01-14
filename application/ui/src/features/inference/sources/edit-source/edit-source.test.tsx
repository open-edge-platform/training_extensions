// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TestProviders } from 'test-utils/render';

import { useConnectSourceToPipeline } from '../../../../hooks/api/pipeline.hook';
import { useSourceMutation } from '../hooks/use-source-mutation.hook';
import { IpCamera } from '../ip-camera/ip-camera.component';
import { getIpCameraInitialConfig, ipCameraBodyFormatter } from '../ip-camera/utils';
import { IPCameraSourceConfig } from '../util';
import { EditSource } from './edit-source.component';

vi.mock('../hooks/use-source-mutation.hook');
vi.mock('../../../../hooks/api/pipeline.hook');

describe('EditIpCamera', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    const updatedConfig: IPCameraSourceConfig = {
        id: 'existing-source-id',
        name: 'Updated Camera',
        stream_url: 'rtsp://192.168.1.201:554/stream',
        source_type: 'ip_camera',
        auth_required: true,
    };

    const renderApp = (mockOnSaved = vi.fn()) => {
        render(
            <TestProviders>
                <EditSource
                    config={getIpCameraInitialConfig()}
                    onSaved={mockOnSaved}
                    onBackToList={vi.fn()}
                    componentFields={(state: IPCameraSourceConfig) => <IpCamera defaultState={state} />}
                    bodyFormatter={ipCameraBodyFormatter}
                />
            </TestProviders>
        );
    };

    it('calls connectToPipelineMutation after successful "Save & Connect" submit', async () => {
        const mockOnSaved = vi.fn();
        const mockSourceMutation = vi.fn().mockResolvedValue(updatedConfig.id);
        const mockConnectToPipeline = vi.fn().mockResolvedValue(undefined);

        vi.mocked(useConnectSourceToPipeline).mockReturnValue(mockConnectToPipeline);
        vi.mocked(useSourceMutation).mockReturnValue(mockSourceMutation);

        renderApp(mockOnSaved);

        const nameInput = screen.getByRole('textbox', { name: /Name/i });
        const streamUrlInput = screen.getByRole('textbox', { name: /Stream Url/i });

        await userEvent.clear(nameInput);
        await userEvent.type(nameInput, updatedConfig.name);
        await userEvent.clear(streamUrlInput);
        await userEvent.type(streamUrlInput, updatedConfig.stream_url);
        await userEvent.click(screen.getByRole('button', { name: /Save & Connect/i }));

        await waitFor(() => {
            expect(mockOnSaved).toHaveBeenCalled();
            expect(mockSourceMutation).toHaveBeenCalled();
            expect(mockConnectToPipeline).toHaveBeenCalledWith(updatedConfig.id);
        });
    });

    it('does not call connectToPipelineMutation after successful "Save" submit', async () => {
        const mockOnSaved = vi.fn();
        const mockSourceMutation = vi.fn().mockResolvedValue(updatedConfig.id);
        const mockConnectToPipeline = vi.fn().mockResolvedValue(undefined);

        vi.mocked(useConnectSourceToPipeline).mockReturnValue(mockConnectToPipeline);
        vi.mocked(useSourceMutation).mockReturnValue(mockSourceMutation);

        renderApp(mockOnSaved);

        const nameInput = screen.getByRole('textbox', { name: /Name/i });
        const streamUrlInput = screen.getByRole('textbox', { name: /Stream Url/i });

        await userEvent.clear(nameInput);
        await userEvent.type(nameInput, updatedConfig.name);
        await userEvent.clear(streamUrlInput);
        await userEvent.type(streamUrlInput, updatedConfig.stream_url);
        await userEvent.click(screen.getByRole('button', { name: /^Save$/i }));

        await waitFor(() => {
            expect(mockOnSaved).toHaveBeenCalled();
            expect(mockSourceMutation).toHaveBeenCalled();
            expect(mockConnectToPipeline).not.toHaveBeenCalled();
        });
    });
});
