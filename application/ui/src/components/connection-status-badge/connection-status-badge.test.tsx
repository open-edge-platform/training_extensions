// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { ConnectionStatusBadge } from './connection-status-badge.component';

describe('ConnectionStatusBadge', () => {
    it('shows "in use" state', () => {
        render(<ConnectionStatusBadge isInUse isUnreachable={false} isPending={false} />);

        const label = screen.getByText(/In use/);

        expect(label).toBeVisible();
    });

    it('shows ready state', () => {
        render(<ConnectionStatusBadge isInUse={false} isUnreachable={false} isPending={false} />);

        expect(screen.getByText(/Ready/)).toBeVisible();
    });

    it('shows testing state when pending', () => {
        render(<ConnectionStatusBadge isInUse={false} isUnreachable={false} isPending />);

        expect(screen.getByText(/Testing connection\.\./)).toBeVisible();
    });

    it('shows API error message when unreachable due to request failure', () => {
        render(
            <ConnectionStatusBadge
                isInUse={false}
                isUnreachable
                isPending={false}
                errorMessage={'Connection timed out'}
            />
        );

        expect(screen.getByText(/Connection timed out/)).toBeVisible();
    });
});
