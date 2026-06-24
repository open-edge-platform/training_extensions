// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { ConnectionStatusBadge } from './connection-status-badge.component';

describe('ConnectionStatusBadge', () => {
    it('shows in use state', () => {
        render(<ConnectionStatusBadge isInUse isUnreachable={false} isPending={false} />);

        expect(screen.getByText('In use')).toBeVisible();
    });

    it('shows unreachable state', () => {
        render(<ConnectionStatusBadge isInUse={false} isUnreachable isPending={false} />);

        expect(screen.getByText('Unreachable')).toBeVisible();
    });

    it('shows reachable state', () => {
        render(<ConnectionStatusBadge isInUse={false} isUnreachable={false} isPending={false} />);

        expect(screen.getByText('Reachable')).toBeVisible();
    });

    it('shows checking state when pending', () => {
        render(<ConnectionStatusBadge isInUse={false} isUnreachable={false} isPending />);

        expect(screen.getByText('Checking')).toBeVisible();
    });
});
