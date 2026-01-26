// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Toast } from '@geti/ui';
import userEvent from '@testing-library/user-event';
import { getMockedLabel } from 'mocks/mock-labels';
import { fireEvent, render, screen } from 'test-utils/render';

import { Label, TaskType } from '../../../constants/shared-types';
import { LabelSelection } from './label-selection.component';

const mockLabels = [getMockedLabel({ id: 'id-1', name: 'Car' })];

const App = ({
    labels = mockLabels,
    setLabels = vi.fn(),
    taskType = 'detection',
}: {
    labels?: Label[];
    setLabels?: () => void;
    taskType?: TaskType;
}) => {
    return (
        <>
            <LabelSelection labels={labels} setLabels={setLabels} taskType={taskType} />
            <Toast />
        </>
    );
};

describe('LabelSelection', () => {
    it('renders initial label item', () => {
        render(<App />);

        expect(screen.getByText('Car')).toBeInTheDocument();
    });

    it('creates a new label item', async () => {
        const mockSetLabels = vi.fn();
        render(<App setLabels={mockSetLabels} />);

        const addButton = screen.getByRole('button', { name: /create label/i });
        const input = screen.getByRole('textbox', { name: 'Create label input' });
        const labelName = 'Object';

        await userEvent.type(input, labelName);

        fireEvent.click(addButton);

        expect(mockSetLabels).toHaveBeenCalledWith(
            expect.arrayContaining([mockLabels[0], expect.objectContaining({ name: labelName })])
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

        expect(screen.getByText('Car')).toBeInTheDocument();
        expect(screen.getByText('People')).toBeInTheDocument();
        expect(mockSetLabels).toHaveBeenCalledWith(expect.arrayContaining([mockLabels[0]]));
    });
});
