// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@testing-library/react';

import { StatusTag } from './status-tag.component';

describe('StatusTag', () => {
    it('renders correct status based on props', () => {
        const { rerender } = render(<StatusTag />);
        expect(screen.getByText('Disconnected')).toBeInTheDocument();

        rerender(<StatusTag isConnected={false} />);
        expect(screen.getByText('Disconnected')).toBeInTheDocument();

        rerender(<StatusTag isConnected={true} />);
        expect(screen.getByText('Connected')).toBeInTheDocument();

        rerender(<StatusTag isError={true} />);
        expect(screen.getByText('Error')).toBeInTheDocument();

        rerender(<StatusTag isConnected={true} isError={true} />);
        expect(screen.getByText('Error')).toBeInTheDocument();
        expect(screen.queryByText('Connected')).not.toBeInTheDocument();
    });
});
