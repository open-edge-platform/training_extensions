// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen } from '@testing-library/react';

import { ParentRevisionModel } from './parent-revision-model.component';

describe('ParentRevisionModel', () => {
    it('renders complete text with link and handles various model names', () => {
        const { container, rerender } = render(<ParentRevisionModel id='123' name='Parent Model' />);

        expect(screen.getByText(/Fine-tuned from/i)).toBeInTheDocument();
        expect(screen.getByText('Parent Model')).toBeInTheDocument();
        expect(container.textContent).toBe('Fine-tuned from Parent Model');

        rerender(<ParentRevisionModel id='2' name='Very Long Parent Model Name That Exceeds Normal Length' />);
        expect(screen.getByText('Very Long Parent Model Name That Exceeds Normal Length')).toBeInTheDocument();

        rerender(<ParentRevisionModel id='3' name='Model-v2.1_final (beta)' />);
        expect(screen.getByText('Model-v2.1_final (beta)')).toBeInTheDocument();
    });

    it('handles onExpandModel callback correctly', () => {
        const mockOnExpandModel = vitest.fn();

        render(<ParentRevisionModel id='model-123' name='Parent Model' onExpandModel={mockOnExpandModel} />);
        fireEvent.click(screen.getByText('Parent Model'));
        expect(mockOnExpandModel).toHaveBeenCalledTimes(1);
        expect(mockOnExpandModel).toHaveBeenCalledWith('model-123');

        fireEvent.click(screen.getByText('Parent Model'));
        fireEvent.click(screen.getByText('Parent Model'));
        expect(mockOnExpandModel).toHaveBeenCalledTimes(3);
    });
});
