// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render } from '@testing-library/react';

import type { Point } from '../../../../../shared/types';
import { CrosshairLine } from './crosshair-line.component';

describe('CrosshairLine', () => {
    const mockPoint: Point = { x: 100, y: 200 };

    it('renders rect with correct direction attributes', () => {
        const { container } = render(
            <svg>
                <CrosshairLine zoom={1} point={mockPoint} direction='horizontal' />
            </svg>
        );
        const hRect = container.querySelector('rect');
        expect(hRect).toBeInTheDocument();
        expect(hRect).toHaveAttribute('y', '200');
        expect(hRect).toHaveAttribute('width', '100%');
        expect(hRect).toHaveAttribute('height', '1');

        const { container: vContainer } = render(
            <svg>
                <CrosshairLine zoom={1} point={mockPoint} direction='vertical' />
            </svg>
        );
        const vRect = vContainer.querySelector('rect');
        expect(vRect).toHaveAttribute('x', '100');
        expect(vRect).toHaveAttribute('width', '1');
        expect(vRect).toHaveAttribute('height', '100%');
    });

    it('handles zoom level 2 correctly', () => {
        const { container } = render(
            <svg>
                <CrosshairLine zoom={2} point={mockPoint} direction='horizontal' />
            </svg>
        );
        expect(container.querySelector('rect')).toHaveAttribute('height', '0.5');
    });

    it('handles high zoom level correctly', () => {
        const { container } = render(
            <svg>
                <CrosshairLine zoom={4} point={mockPoint} direction='vertical' />
            </svg>
        );
        const rect = container.querySelector('rect');
        expect(rect).toHaveAttribute('width', '0.25');
        expect(rect).toHaveAttribute('stroke-width', '0.25');
    });

    it('handles low zoom level correctly', () => {
        const { container } = render(
            <svg>
                <CrosshairLine zoom={0.5} point={mockPoint} direction='horizontal' />
            </svg>
        );
        expect(container.querySelector('rect')).toHaveAttribute('height', '2');
    });

    it('applies correct styling', () => {
        const { container } = render(
            <svg>
                <CrosshairLine zoom={1} point={mockPoint} direction='horizontal' />
            </svg>
        );
        const rect = container.querySelector('rect');
        expect(rect).toHaveAttribute('fill', 'white');
        expect(rect).toHaveAttribute('fill-opacity', '0.9');
        expect(rect).toHaveAttribute('stroke', '#000000');
        expect(rect).toHaveAttribute('stroke-opacity', '0.12');
        expect(rect).toHaveAttribute('stroke-width', '1');
    });

    it('handles different integer point coordinates', () => {
        const differentPoint: Point = { x: 50, y: 75 };
        const { container } = render(
            <svg>
                <CrosshairLine zoom={1} point={differentPoint} direction='horizontal' />
            </svg>
        );
        expect(container.querySelector('rect')).toHaveAttribute('y', '75');
    });

    it('handles decimal point coordinates', () => {
        const decimalPoint: Point = { x: 10.5, y: 20.75 };
        const { container } = render(
            <svg>
                <CrosshairLine zoom={1} point={decimalPoint} direction='vertical' />
            </svg>
        );
        expect(container.querySelector('rect')).toHaveAttribute('x', '10.5');
    });
});
