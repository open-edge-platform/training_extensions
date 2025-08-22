// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Toast } from '@geti/ui';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';

import { LabelSelection } from './label-selection.component';

const renderLabelSelection = () => {
    return render(
        <>
            <LabelSelection />
            <Toast />
        </>
    );
};

describe('LabelSelection', () => {
    it('renders initial label item', () => {
        renderLabelSelection();

        expect(screen.getByDisplayValue('Car')).toBeInTheDocument();
    });

    it('adds a new label item when "Add next object" is clicked', () => {
        renderLabelSelection();

        const addButton = screen.getByRole('button', { name: /add next object/i });
        fireEvent.click(addButton);

        expect(screen.getByLabelText('Label input for Car')).toBeInTheDocument();
        expect(screen.getByLabelText('Label input for Object')).toBeInTheDocument();
    });

    it('deletes a label item when delete is clicked', () => {
        renderLabelSelection();

        const addButton = screen.getByRole('button', { name: /add next object/i });
        fireEvent.click(addButton);

        const deleteButtonObject = screen.getByRole('button', { name: /delete label object/i });
        fireEvent.click(deleteButtonObject);

        expect(screen.getByLabelText('Label input for Car')).toBeInTheDocument();
        expect(screen.queryByLabelText('Label input for Object')).not.toBeInTheDocument();
    });

    it('does not delete the last remaining label item', async () => {
        renderLabelSelection();

        const deleteButton = screen.getByRole('button', { name: /delete label car/i });
        fireEvent.click(deleteButton);

        expect(screen.getByLabelText('Label input for Car')).toBeInTheDocument();

        await waitFor(() => {
            expect(screen.getByText('At least one object is required')).toBeInTheDocument();
        });
    });

    it('can edit the label name', () => {
        renderLabelSelection();

        const input = screen.getByLabelText('Label input for Car');
        fireEvent.change(input, { target: { value: 'Truck' } });

        expect(screen.getByLabelText('Label input for Truck')).toBeInTheDocument();
    });
});
