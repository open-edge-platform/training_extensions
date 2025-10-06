// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Toast } from '@geti/ui';
import { getMockedLabel } from 'mocks/mock-labels';
import { fireEvent, render, screen, waitFor } from 'test-utils/render';

import { LabelSelection } from './label-selection.component';

const mockLabels = [getMockedLabel({ id: 'id-1', name: 'Car' })];

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
            expect.arrayContaining([mockLabels[0], expect.objectContaining({ name: 'Object' })])
        );
    });

    it('deletes a label item when delete is clicked', () => {
        const mockSetLabels = vi.fn();
        render(
            <App
                labels={[mockLabels[0], getMockedLabel({ id: 'id-2', color: '#F20004', name: 'People' })]}
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
        const mockSetLabels = vi.fn();
        render(<App setLabels={mockSetLabels} />);

        const input = screen.getByLabelText('Label input for Car');
        fireEvent.change(input, { target: { value: 'Truck' } });

        expect(mockSetLabels).toHaveBeenCalledWith(
            expect.arrayContaining([
                expect.objectContaining({
                    id: 'id-1',
                    name: 'Truck',
                    color: expect.any(String),
                }),
            ])
        );
    });
});
