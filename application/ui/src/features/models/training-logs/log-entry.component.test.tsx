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

    it.each(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'SUCCESS', 'CRITICAL'] as LogLevel[])(
        'applies the correct color for the %s level',
        (level) => {
            const entry = getMockedLogEntry({
                level: { icon: '', name: level, no: 0 },
                message: `${level} message`,
            });

            render(<LogEntry entry={entry} />);

            const levelElement = screen.getByText(level);
            expect(levelElement).toHaveStyle({ color: LOG_LEVEL_COLORS[level] });
        }
    );
});
