// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Toast } from '@geti/ui';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';

import { TestProviders } from '../../../providers';
import { ProjectProvider } from '../project-provider.component';
import { LabelSelection } from './label-selection.component';

const App = () => {
    return (
        <TestProviders>
            <ProjectProvider>
                <>
                    <LabelSelection />
                    <Toast />
                </>
            </ProjectProvider>
        </TestProviders>
    );
};

describe('LabelSelection', () => {
    it('renders initial label item', () => {
        render(<App />);

        expect(screen.getByDisplayValue('Car')).toBeInTheDocument();
    });

    it('adds a new label item when "Add next object" is clicked', () => {
        render(<App />);

        const addButton = screen.getByRole('button', { name: /add next object/i });
        fireEvent.click(addButton);

        expect(screen.getByLabelText('Label input for Car')).toBeInTheDocument();
        expect(screen.getByLabelText('Label input for Object')).toBeInTheDocument();
    });

    it('deletes a label item when delete is clicked', () => {
        render(<App />);

        const addButton = screen.getByRole('button', { name: /add next object/i });
        fireEvent.click(addButton);

        const deleteButtonObject = screen.getByRole('button', { name: /delete label object/i });
        fireEvent.click(deleteButtonObject);

        expect(screen.getByLabelText('Label input for Car')).toBeInTheDocument();
        expect(screen.queryByLabelText('Label input for Object')).not.toBeInTheDocument();
    });

    it('does not delete the last remaining label item', async () => {
        render(<App />);

        const deleteButton = screen.getByRole('button', { name: /delete label car/i });
        fireEvent.click(deleteButton);

        expect(screen.getByLabelText('Label input for Car')).toBeInTheDocument();

        await waitFor(() => {
            expect(screen.getByText('At least one object is required')).toBeInTheDocument();
        });
    });

    it('can edit the label name', () => {
        render(<App />);

        const input = screen.getByLabelText('Label input for Car');
        fireEvent.change(input, { target: { value: 'Truck' } });

        expect(screen.getByLabelText('Label input for Truck')).toBeInTheDocument();
    });
});
