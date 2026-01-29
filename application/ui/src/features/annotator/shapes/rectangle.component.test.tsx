// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render } from '@testing-library/react';

import type { Rect as RectInterface } from '../../../shared/types';
import { Rectangle } from './rectangle.component';

describe('Rectangle', () => {
    const mockRect: RectInterface = { type: 'rectangle', x: 10, y: 20, width: 100, height: 50 };

    const mockStyles: React.SVGProps<SVGRectElement> = {
        fill: 'blue',
        stroke: 'red',
        strokeWidth: 2,
    };

    it('renders rect with correct attributes and styles', () => {
        const { container } = render(
            <svg>
                <Rectangle rect={mockRect} styles={mockStyles} ariaLabel='test rectangle' />
            </svg>
        );

        const rect = container.querySelector('rect');
        expect(rect).toBeInTheDocument();
        expect(rect).toHaveAttribute('x', '10');
        expect(rect).toHaveAttribute('y', '20');
        expect(rect).toHaveAttribute('width', '100');
        expect(rect).toHaveAttribute('height', '50');
        expect(rect).toHaveAttribute('fill', 'blue');
        expect(rect).toHaveAttribute('stroke', 'red');
        expect(rect).toHaveAttribute('stroke-width', '2');
        expect(rect).toHaveAttribute('aria-label', 'test rectangle');
    });

    it('handles edge cases for coordinates and dimensions', () => {
        const zeroRect: RectInterface = { type: 'rectangle', x: 0, y: 0, width: 0, height: 0 };
        const { container: zeroContainer } = render(
            <svg>
                <Rectangle rect={zeroRect} styles={mockStyles} ariaLabel='zero rect' />
            </svg>
        );
        const zeroRectEl = zeroContainer.querySelector('rect');
        expect(zeroRectEl).toHaveAttribute('x', '0');
        expect(zeroRectEl).toHaveAttribute('width', '0');

        const negativeRect: RectInterface = { type: 'rectangle', x: -10, y: -20, width: 100, height: 50 };
        const { container: negContainer } = render(
            <svg>
                <Rectangle rect={negativeRect} styles={mockStyles} ariaLabel='negative rect' />
            </svg>
        );
        const negRectEl = negContainer.querySelector('rect');
        expect(negRectEl).toHaveAttribute('x', '-10');
        expect(negRectEl).toHaveAttribute('y', '-20');

        const decimalRect: RectInterface = { type: 'rectangle', x: 10.5, y: 20.75, width: 100.25, height: 50.5 };
        const { container: decContainer } = render(
            <svg>
                <Rectangle rect={decimalRect} styles={mockStyles} ariaLabel='decimal rect' />
            </svg>
        );
        const decRectEl = decContainer.querySelector('rect');
        expect(decRectEl).toHaveAttribute('x', '10.5');
        expect(decRectEl).toHaveAttribute('y', '20.75');
    });

    it('spreads additional SVG props from styles', () => {
        const extendedStyles: React.SVGProps<SVGRectElement> = {
            fill: 'green',
            opacity: 0.5,
            className: 'custom-class',
        };

        const { container } = render(
            <svg>
                <Rectangle rect={mockRect} styles={extendedStyles} ariaLabel='extended rect' />
            </svg>
        );

        const rect = container.querySelector('rect');
        expect(rect).toHaveAttribute('opacity', '0.5');
        expect(rect).toHaveClass('custom-class');
    });
});
