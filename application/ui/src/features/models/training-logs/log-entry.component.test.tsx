// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { getMockedLogEntry } from 'mocks/mock-log-entry';
import { render } from 'test-utils/render';

import { LogEntry } from './log-entry.component';
import { LOG_LEVEL_COLORS, type LogLevel } from './log-types';

describe('LogEntry', () => {
    it('renders timestamp, level, source, and message', () => {
        const entry = getMockedLogEntry({
            message: 'Starting the training loop',
            level: { icon: 'ℹ️', name: 'INFO', no: 20 },
            time: { repr: '2026-02-06 10:30:01.000000+00:00', timestamp: 1770328201.0 },
            name: 'otx_trainer',
            function: 'train_model',
            line: 42,
        });

        render(<LogEntry entry={entry} />);

        const listitem = screen.getByRole('listitem');
        expect(listitem).toBeInTheDocument();

        expect(screen.getByText('INFO')).toBeInTheDocument();
        expect(screen.getByText('Starting the training loop')).toBeInTheDocument();
        expect(screen.getByText('otx_trainer:train_model:42')).toBeInTheDocument();
    });

    it('applies the correct color for each log level', () => {
        const levels: LogLevel[] = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'SUCCESS', 'CRITICAL'];

        levels.forEach((level) => {
            const entry = getMockedLogEntry({
                level: { icon: '', name: level, no: 0 },
                message: `${level} message`,
            });

            const { unmount } = render(<LogEntry entry={entry} />);

            const levelElement = screen.getByText(level);
            expect(levelElement).toHaveStyle({ color: LOG_LEVEL_COLORS[level] });

            unmount();
        });
    });

    it('omits the timestamp, source when both are empty', () => {
        const entry = getMockedLogEntry({
            message: 'No timestamp message',
            time: { repr: '', timestamp: 0 },
            name: '',
            function: '',
        });

        render(<LogEntry entry={entry} />);

        expect(screen.getByText('No timestamp message')).toBeInTheDocument();
        // No timestamp, no source → only level + message should be rendered
        expect(screen.getByRole('listitem').children).toHaveLength(2);
    });

    it('renders source without line number when line is 0', () => {
        const entry = getMockedLogEntry({
            message: 'Test message',
            name: 'test_module',
            function: 'test_func',
            line: 0,
        });

        render(<LogEntry entry={entry} />);

        expect(screen.getByText('test_module:test_func')).toBeInTheDocument();
        expect(screen.queryByText(/:0$/)).not.toBeInTheDocument();
    });

    it('renders source with only name when function is empty', () => {
        const entry = getMockedLogEntry({
            message: 'Test message',
            name: 'test_module',
            function: '',
            line: 0,
        });

        render(<LogEntry entry={entry} />);

        expect(screen.getByText('test_module')).toBeInTheDocument();
    });

    it('renders source with only function when name is empty', () => {
        const entry = getMockedLogEntry({
            message: 'Test message',
            name: '',
            function: 'test_func',
            line: 10,
        });

        render(<LogEntry entry={entry} />);

        expect(screen.getByText('test_func:10')).toBeInTheDocument();
    });

    it('trims whitespace from the message', () => {
        const entry = getMockedLogEntry({
            message: '  Padded message  ',
        });

        render(<LogEntry entry={entry} />);

        expect(screen.getByText('Padded message')).toBeInTheDocument();
    });
});
