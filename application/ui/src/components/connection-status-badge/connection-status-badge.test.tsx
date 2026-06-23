// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { ConnectionStatusBadge } from './connection-status-badge.component';

describe('ConnectionStatusBadge', () => {
    beforeEach(() => {
        vi.useFakeTimers();
        vi.setSystemTime(new Date('2026-01-01T10:00:00.000Z'));
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('shows available with last checked time', () => {
        render(<ConnectionStatusBadge isAvailable lastCheckedAt={new Date('2026-01-01T09:30:00.000Z').valueOf()} />);

        expect(screen.getByText(/Available/)).toBeVisible();
        expect(screen.getByText('Last checked: 30 minutes ago')).toBeVisible();
    });

    it('shows unavailable with last checked time', () => {
        render(
            <ConnectionStatusBadge isAvailable={false} lastCheckedAt={new Date('2026-01-01T09:59:00.000Z').valueOf()} />
        );

        expect(screen.getByText(/Unavailable/)).toBeVisible();
        expect(screen.getByText('Last checked: a minute ago')).toBeVisible();
    });
});
