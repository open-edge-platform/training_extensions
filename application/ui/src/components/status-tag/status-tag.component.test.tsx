// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@testing-library/react';

import { StatusTag } from './status-tag.component';

describe('StatusTag', () => {
    it('renders Disconnected by default', () => {
        render(<StatusTag />);
        expect(screen.getByText('Disconnected')).toBeInTheDocument();
    });

    it('renders Disconnected when isConnected is false', () => {
        render(<StatusTag isConnected={false} />);
        expect(screen.getByText('Disconnected')).toBeInTheDocument();
    });

    it('renders Connected when isConnected is true', () => {
        render(<StatusTag isConnected={true} />);
        expect(screen.getByText('Connected')).toBeInTheDocument();
    });

    it('renders Error when isError is true', () => {
        render(<StatusTag isError={true} />);
        expect(screen.getByText('Error')).toBeInTheDocument();
    });

    it('renders Error when both isConnected and isError are true', () => {
        render(<StatusTag isConnected={true} isError={true} />);
        expect(screen.getByText('Error')).toBeInTheDocument();
        expect(screen.queryByText('Connected')).not.toBeInTheDocument();
    });
});
