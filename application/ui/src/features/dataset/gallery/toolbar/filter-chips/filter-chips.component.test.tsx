// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { FilterChips } from './filter-chips.component';

import classes from './filter-chips.module.scss';

const getCloseIcon = (container: HTMLElement): SVGElement => {
    const icon = container.querySelector(`svg.${classes.closeIcon}`);

    if (icon === null) {
        throw new Error('Close icon not found');
    }

    return icon as SVGElement;
};

describe('FilterChips', () => {
    it('renders the given name', () => {
        render(<FilterChips name={'Cat'} onClose={vi.fn()} />);

        expect(screen.getByText('Cat')).toBeVisible();
    });

    it('calls onClose when the close icon is clicked', () => {
        const mockOnClose = vi.fn();
        const { container } = render(<FilterChips name={'Cat'} onClose={mockOnClose} />);

        fireEvent.click(getCloseIcon(container));

        expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it('does not propagate the click event when closing', () => {
        const mockOnClose = vi.fn();
        const mockOnContainerClick = vi.fn();

        const { container } = render(
            <section onClick={mockOnContainerClick}>
                <FilterChips name={'Cat'} onClose={mockOnClose} />
            </section>
        );

        fireEvent.click(getCloseIcon(container));

        expect(mockOnClose).toHaveBeenCalledTimes(1);
        expect(mockOnContainerClick).not.toHaveBeenCalled();
    });
});
