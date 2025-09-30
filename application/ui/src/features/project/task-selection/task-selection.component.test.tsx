// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@test-utils/render';
import userEvent from '@testing-library/user-event';

import { TaskType } from './interface';
import { TaskSelection } from './task-selection.component';

describe('TaskSelection', () => {
    const App = ({ selectedTask = 'detection', setSelectedTask = vi.fn() }) => {
        return <TaskSelection selectedTask={selectedTask as TaskType} setSelectedTask={setSelectedTask} />;
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

    it('selects a task when the whole element is clicked', async () => {
        const mockSetSelectedTask = vi.fn();
        render(<App setSelectedTask={mockSetSelectedTask} />);

        const segOption = screen.getByLabelText('Task option: Image Segmentation');
        await userEvent.click(segOption);

        expect(mockSetSelectedTask).toHaveBeenCalledWith('instance_segmentation');
    });

    it('selects a task when the radio element is clicked', async () => {
        const mockSetSelectedTask = vi.fn();
        render(<App setSelectedTask={mockSetSelectedTask} />);

        const classRadio = screen.getByLabelText('classification');
        await userEvent.click(classRadio);

        expect(mockSetSelectedTask).toHaveBeenCalledWith('classification');
    });
});
