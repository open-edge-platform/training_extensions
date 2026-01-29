// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen } from '@testing-library/react';

import { IconWrapper } from './icon-wrapper.component';

describe('IconWrapper', () => {
    it('renders children and props correctly', () => {
        const { container } = render(
            <IconWrapper isSelected={true}>
                <span data-testid='test-icon'>Icon</span>
            </IconWrapper>
        );

        expect(screen.getByTestId('test-icon')).toBeInTheDocument();
        expect(screen.getByText('Icon')).toBeInTheDocument();
        expect(container.firstChild).toBeInTheDocument();
    });

    it('handles onPress events correctly', () => {
        const mockOnPress = vitest.fn();

        render(
            <IconWrapper onPress={mockOnPress} isSelected={true}>
                <span>Icon</span>
            </IconWrapper>
        );
        fireEvent.click(screen.getByText('Icon'));
        expect(mockOnPress).toHaveBeenCalledTimes(1);

        render(
            <IconWrapper>
                <span>No callback</span>
            </IconWrapper>
        );
        expect(() => fireEvent.click(screen.getByText('No callback'))).not.toThrow();
    });
});
