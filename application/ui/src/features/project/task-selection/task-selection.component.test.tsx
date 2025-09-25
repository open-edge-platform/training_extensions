// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen } from '@testing-library/react';

import { TestProviders } from '../../../providers';
import { TaskType } from './interface';
import { TaskSelection } from './task-selection.component';

describe('TaskSelection', () => {
    const App = ({ selectedTask = 'detection', setSelectedTask = vi.fn() }) => {
        return (
            <TestProviders>
                <TaskSelection selectedTask={selectedTask as TaskType} setSelectedTask={setSelectedTask} />
            </TestProviders>
        );
    };

    it('renders all task options', () => {
        render(<App />);

        expect(screen.getByLabelText('Task option: Object Detection')).toBeInTheDocument();
        expect(screen.getByLabelText('Task option: Image Segmentation')).toBeInTheDocument();
        expect(screen.getByLabelText('Task option: Image Classification')).toBeInTheDocument();
    });

    it('selects the first task by default', () => {
        render(<App />);

        const radio = screen.getByLabelText('detection');
        expect(radio).toBeChecked();
    });

    it('selects a task when the whole element is clicked', () => {
        const mockSetSelectedTask = vi.fn();
        render(<App setSelectedTask={mockSetSelectedTask} />);

        const segOption = screen.getByLabelText('Task option: Image Segmentation');
        fireEvent.click(segOption);

        expect(mockSetSelectedTask).toHaveBeenCalledWith('segmentation');
    });

    it('selects a task when the radio element is clicked', () => {
        const mockSetSelectedTask = vi.fn();
        render(<App setSelectedTask={mockSetSelectedTask} />);

        const classRadio = screen.getByLabelText('classification');
        fireEvent.click(classRadio);

        expect(mockSetSelectedTask).toHaveBeenCalledWith('classification');
    });
});
