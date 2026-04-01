// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMockLogs } from 'mocks/mock-log-entry';
import { render } from 'test-utils/render';

import { LogViewer } from './log-viewer.component';

describe('LogViewer', () => {
    it('filters logs by level when level picker is changed', async () => {
        const logs = createMockLogs();

        render(<LogViewer logs={logs} />);

        expect(screen.getByText('4 / 5 entries')).toBeInTheDocument();

        const pickerButton = screen.getByRole('button', { name: /minimum log level/i });
        await userEvent.click(pickerButton);

        const warningOption = await screen.findByRole('option', { name: /WARNING/i });
        await userEvent.click(warningOption);

        expect(screen.getByText('2 / 5 entries')).toBeInTheDocument();
    });

    it('applies both level and search filters together', async () => {
        const logs = createMockLogs();

        render(<LogViewer logs={logs} />);

        const pickerButton = screen.getByRole('button', { name: /minimum log level/i });
        await userEvent.click(pickerButton);

        const debugOption = await screen.findByRole('option', { name: /DEBUG/i });
        await userEvent.click(debugOption);

        expect(screen.getByText('5 / 5 entries')).toBeInTheDocument();

        const searchField = screen.getByLabelText('Search logs');
        await userEvent.type(searchField, 'memory');

        expect(screen.getByText('1 / 5 entries')).toBeInTheDocument();
        expect(screen.getByText('Low GPU memory detected')).toBeInTheDocument();
    });

    it('shows "No log entries" when logs array is empty', () => {
        render(<LogViewer logs={[]} />);

        expect(screen.getByText('No log entries')).toBeInTheDocument();
        expect(screen.getByText('0 / 0 entries')).toBeInTheDocument();
    });

    it('shows "No matching log entries" when filters exclude all logs', async () => {
        const logs = createMockLogs();

        render(<LogViewer logs={logs} />);

        const searchField = screen.getByLabelText('Search logs');
        await userEvent.type(searchField, 'nonexistent query xyz');

        expect(screen.getByText('No matching log entries')).toBeInTheDocument();
        expect(screen.getByText('0 / 5 entries')).toBeInTheDocument();
    });

    it('shows contextual help tooltip for the level picker', async () => {
        const logs = createMockLogs();

        render(<LogViewer logs={logs} />);

        const helpButton = screen.getByRole('button', { name: /level information/i });
        await userEvent.click(helpButton);

        expect(await screen.findByText('Minimum log level')).toBeInTheDocument();
        expect(await screen.findByText(/shows log entries at the selected level and above/i)).toBeInTheDocument();
    });

    it('displays connection status indicator when connectionStatus is provided', () => {
        const logs = createMockLogs();

        const { rerender } = render(<LogViewer logs={logs} isStreaming connectionStatus={'connected'} />);
        expect(screen.getByText('Live')).toBeInTheDocument();
        rerender(<LogViewer logs={logs} isStreaming connectionStatus={'connecting'} />);
        expect(screen.getByText('Connecting...')).toBeInTheDocument();
        rerender(<LogViewer logs={logs} isStreaming connectionStatus={'disconnected'} />);
        expect(screen.getByText('Disconnected')).toBeInTheDocument();

        rerender(<LogViewer logs={logs} />);
        expect(screen.queryByText('Live')).not.toBeInTheDocument();
        expect(screen.queryByText('Connecting...')).not.toBeInTheDocument();
        expect(screen.queryByText('Disconnected')).not.toBeInTheDocument();
    });
});
