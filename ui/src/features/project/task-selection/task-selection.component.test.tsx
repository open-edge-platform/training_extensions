// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen } from '@testing-library/react';

import { TestProviders } from '../../../providers';
import { TaskSelection } from './task-selection.component';

describe('TaskSelection', () => {
    const App = () => {
        return (
            <TestProviders>
                <TaskSelection selectedTask={'detection'} setSelectedTask={vi.fn()} />
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
        render(<App />);

        const segOption = screen.getByLabelText('Task option: Image Segmentation');
        fireEvent.click(segOption);

        const segRadio = screen.getByLabelText('segmentation');
        expect(segRadio).toBeChecked();
    });

    it('selects a task when the radio element is clicked', () => {
        render(<App />);

        const classRadio = screen.getByLabelText('classification');
        fireEvent.click(classRadio);

        expect(classRadio).toBeChecked();
    });

    it('only one task is selected at a time', () => {
        render(<App />);

        const segOption = screen.getByLabelText('Task option: Image Segmentation');
        fireEvent.click(segOption);

        expect(screen.getByLabelText('segmentation')).toBeChecked();
        expect(screen.getByLabelText('detection')).not.toBeChecked();
        expect(screen.getByLabelText('classification')).not.toBeChecked();
    });
});
