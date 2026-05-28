// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from 'test-utils/render';

import { useConnectSinkToPipeline } from '../../../../hooks/api/pipeline.hook';
import { useSinkMutation } from '../hooks/use-sink-mutation.hook';
import { LocalFolder } from '../local-folder/local-folder.component';
import { getLocalFolderInitialConfig, localFolderBodyFormatter } from '../local-folder/utils';
import { LocalFolderSinkConfig, OutputFormat } from '../utils';
import { EditSink } from './edit-sink.component';

vi.mock('../hooks/use-sink-mutation.hook');
vi.mock('../../../../hooks/api/pipeline.hook');

describe('EditSink', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    const updatedConfig: LocalFolderSinkConfig = {
        id: 'existing-sink-id',
        name: 'Updated folder sink',
        sink_type: 'folder',
        rate_limit: null,
        folder_path: '/data/output',
        output_formats: [],
    };

    const renderApp = (mockOnSaved = vi.fn(), isConnected = false) => {
        render(
            <EditSink
                config={{ ...getLocalFolderInitialConfig(), output_formats: [OutputFormat.PREDICTIONS] }}
                onSaved={mockOnSaved}
                onBackToList={vi.fn()}
                componentFields={(state: LocalFolderSinkConfig) => <LocalFolder defaultState={state} />}
                bodyFormatter={localFolderBodyFormatter}
                isConnected={isConnected}
            />
        );
    };

    it('calls connectToPipelineMutation after successful "Save & Connect" submit', async () => {
        const mockOnSaved = vi.fn();
        const mockSinkMutation = vi.fn().mockResolvedValue(updatedConfig.id);
        const mockConnectToPipeline = vi.fn().mockResolvedValue(undefined);

        vi.mocked(useConnectSinkToPipeline).mockReturnValue(mockConnectToPipeline);
        vi.mocked(useSinkMutation).mockReturnValue(mockSinkMutation);

        renderApp(mockOnSaved);

        const folderPathInput = screen.getByRole('textbox', { name: /Folder Path/i });
        await userEvent.clear(folderPathInput);
        await userEvent.type(folderPathInput, updatedConfig.folder_path);
        await userEvent.click(screen.getByRole('button', { name: /Save & Connect/i }));

        await waitFor(() => {
            expect(mockOnSaved).toHaveBeenCalled();
            expect(mockSinkMutation).toHaveBeenCalled();
            expect(mockConnectToPipeline).toHaveBeenCalledWith(updatedConfig.id);
        });
    });

    it('does not call connectToPipelineMutation after successful "Save" submit', async () => {
        const mockOnSaved = vi.fn();
        const mockSinkMutation = vi.fn().mockResolvedValue(updatedConfig.id);
        const mockConnectToPipeline = vi.fn().mockResolvedValue(undefined);

        vi.mocked(useConnectSinkToPipeline).mockReturnValue(mockConnectToPipeline);
        vi.mocked(useSinkMutation).mockReturnValue(mockSinkMutation);

        renderApp(mockOnSaved);

        const folderPathInput = screen.getByRole('textbox', { name: /Folder Path/i });
        await userEvent.clear(folderPathInput);
        await userEvent.type(folderPathInput, updatedConfig.folder_path);
        await userEvent.click(screen.getByRole('button', { name: 'Save' }));

        await waitFor(() => {
            expect(mockOnSaved).toHaveBeenCalled();
            expect(mockSinkMutation).toHaveBeenCalled();
            expect(mockConnectToPipeline).not.toHaveBeenCalled();
        });
    });

    it('hides "Save & Connect" button when already connected', () => {
        vi.mocked(useConnectSinkToPipeline).mockReturnValue(vi.fn());
        vi.mocked(useSinkMutation).mockReturnValue(vi.fn());

        renderApp(vi.fn(), true);

        expect(screen.queryByRole('button', { name: /Save & Connect/i })).not.toBeInTheDocument();
    });

    it('shows "Save & Connect" button when not connected', () => {
        vi.mocked(useConnectSinkToPipeline).mockReturnValue(vi.fn());
        vi.mocked(useSinkMutation).mockReturnValue(vi.fn());

        renderApp(vi.fn(), false);

        expect(screen.getByRole('button', { name: /Save & Connect/i })).toBeInTheDocument();
    });
});
