// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen } from '@testing-library/react';

import { CheckboxInput } from './checkbox-input.component';

describe('CheckboxInput', () => {
    it('renders with correct attributes', () => {
        render(<CheckboxInput name='test-checkbox' />);

        const checkbox = screen.getByRole('checkbox', { name: 'test-checkbox' });
        expect(checkbox).toBeInTheDocument();
        expect(checkbox).toHaveAttribute('name', 'test-checkbox');
        expect(checkbox).toHaveAttribute('aria-label', 'test-checkbox');
    });

    it('handles checked state correctly', () => {
        const { rerender } = render(<CheckboxInput name='test-checkbox' />);
        const checkbox = screen.getByRole('checkbox', { name: 'test-checkbox' });
        expect(checkbox).not.toBeChecked();

        rerender(<CheckboxInput name='test-checkbox' isChecked={true} />);
        expect(checkbox).toBeChecked();

        rerender(<CheckboxInput name='test-checkbox' isChecked={false} />);
        expect(checkbox).not.toBeChecked();
    });

    it('calls onChange with correct values when toggled', () => {
        const mockOnChange = vitest.fn();
        const { rerender } = render(<CheckboxInput name='test-checkbox' onChange={mockOnChange} />);

        const checkbox = screen.getByRole('checkbox', { name: 'test-checkbox' });
        fireEvent.click(checkbox);
        expect(mockOnChange).toHaveBeenCalledWith(true);

        rerender(<CheckboxInput name='test-checkbox' isChecked={true} onChange={mockOnChange} />);
        fireEvent.click(checkbox);
        expect(mockOnChange).toHaveBeenCalledWith(false);

        expect(mockOnChange).toHaveBeenCalledTimes(2);
    });

    it('renders as readOnly and works without onChange callback', () => {
        render(<CheckboxInput name='readonly-checkbox' isReadOnly={true} />);
        const readOnlyCheckbox = screen.getByRole('checkbox', { name: 'readonly-checkbox' });
        expect(readOnlyCheckbox).toHaveAttribute('readOnly');

        render(<CheckboxInput name='no-callback' />);
        const checkbox = screen.getByRole('checkbox', { name: 'no-callback' });
        expect(() => fireEvent.click(checkbox)).not.toThrow();
    });
});
