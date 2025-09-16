// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Toast } from '@geti/ui';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';

import { LabelSelection } from './label-selection.component';

const mockLabels = [{ id: 'id-1', colorValue: '#F20004', nameValue: 'Car' }];

const App = ({ labels = mockLabels, setLabels = vi.fn() }) => {
    return (
        <>
            <LabelSelection labels={labels} setLabels={setLabels} />
            <Toast />
        </>
    );
};

describe('LabelSelection', () => {
    it('renders initial label item', () => {
        render(<App />);

        expect(screen.getByDisplayValue('Car')).toBeInTheDocument();
    });

    it('adds a new label item when "Add next object" is clicked', () => {
        const mockSetLabels = vi.fn();
        render(<App setLabels={mockSetLabels} />);

        const addButton = screen.getByRole('button', { name: /add next object/i });
        fireEvent.click(addButton);

        expect(screen.getByLabelText('Label input for Car')).toBeInTheDocument();
        expect(mockSetLabels).toHaveBeenCalledWith(
            expect.arrayContaining([mockLabels[0], expect.objectContaining({ nameValue: 'Object' })])
        );
    });

    it('deletes a label item when delete is clicked', () => {
        const mockSetLabels = vi.fn();
        render(
            <App
                labels={[mockLabels[0], { id: 'id-2', colorValue: '#F20004', nameValue: 'People' }]}
                setLabels={mockSetLabels}
            />
        );

        const deleteButtonObject = screen.getByRole('button', { name: /delete label people/i });
        fireEvent.click(deleteButtonObject);

        expect(screen.getByLabelText('Label input for Car')).toBeInTheDocument();
        expect(mockSetLabels).toHaveBeenCalledWith(expect.arrayContaining([mockLabels[0]]));
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
