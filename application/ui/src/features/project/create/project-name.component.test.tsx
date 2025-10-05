// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen } from 'test-utils/render';

import { ProjectName } from './project-name.component';

describe('ProjectName', () => {
    it('renders initial name', () => {
        render(<ProjectName name='Initial Project' setName={vi.fn()} />);

        expect(screen.getByRole('button', { name: /Initial Project/i })).toBeInTheDocument();
    });

    it('triggers dialog with initial name', () => {
        render(<ProjectName name='Initial Project' setName={vi.fn()} />);

        fireEvent.click(screen.getByRole('button', { name: /Initial Project/i }));

        expect(screen.getByText('Edit Project Name')).toBeInTheDocument();
        expect(screen.getByLabelText('edit project name')).toBeInTheDocument();
    });

    it('updates name onChange', () => {
        const mockSetName = vi.fn();
        render(<ProjectName name='Initial Project' setName={mockSetName} />);

        fireEvent.click(screen.getByRole('button', { name: /Initial Project/i }));

        expect(screen.getByText('Edit Project Name')).toBeInTheDocument();
        expect(screen.getByLabelText('edit project name')).toBeInTheDocument();

        fireEvent.change(screen.getByLabelText('edit project name'), { target: { value: 'New cool name' } });

        expect(mockSetName).toHaveBeenCalledWith('New cool name');
    });
});
