// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedLabel } from 'mocks/mock-labels';
import { render } from 'test-utils/render';
import { describe, expect, it, vi } from 'vitest';

import { Label } from '../types';
import { AnnotationLabels } from './annotation-labels.component';

const mockZoom = { scale: 1, maxZoomIn: 10, hasAnimation: false, translate: { x: 0, y: 0 } };

vi.mock('src/components/zoom/zoom.provider', () => ({
    useZoom: () => mockZoom,
}));

describe('AnnotationLabels', () => {
    const mockOnRemove = vi.fn();

    afterEach(() => {
        mockOnRemove.mockClear();
    });

    it('renders placeholder when no labels provided', () => {
        const { container } = render(
            <svg>
                <AnnotationLabels labels={[]} onRemove={mockOnRemove} />
            </svg>
        );

        const text = container.querySelector('text');
        expect(text).toBeInTheDocument();
        expect(text?.textContent).toBe('No label');
    });

    it('renders single label with name and color', () => {
        const label = getMockedLabel({ name: 'Person', color: '#FF0000' });

        const { container } = render(
            <svg>
                <AnnotationLabels labels={[label]} onRemove={mockOnRemove} />
            </svg>
        );

        const text = container.querySelector('text');
        expect(text?.textContent).toBe('Person');

        const rect = container.querySelector('rect');
        expect(rect).toHaveAttribute('fill', '#FF0000');
    });

    it('renders multiple labels horizontally', () => {
        const labels: Label[] = [
            getMockedLabel({ id: '1', name: 'Person', color: '#FF0000' }),
            getMockedLabel({ id: '2', name: 'Car', color: '#00FF00' }),
        ];

        const { container } = render(
            <svg>
                <AnnotationLabels labels={labels} onRemove={mockOnRemove} />
            </svg>
        );

        const texts = container.querySelectorAll('text');
        expect(texts).toHaveLength(4); // 2 labels + 2 close buttons (x)

        // Check label names
        expect(texts[0]?.textContent).toBe('Person');
        expect(texts[2]?.textContent).toBe('Car');

        // Check close buttons
        expect(texts[1]?.textContent).toBe('x');
        expect(texts[3]?.textContent).toBe('x');
    });

    it('calls onRemove when close button clicked', () => {
        const label = getMockedLabel({ id: 'label-1', name: 'Person' });

        const { container } = render(
            <svg>
                <AnnotationLabels labels={[label]} onRemove={mockOnRemove} />
            </svg>
        );

        const closeButtonGroup = container.querySelector('g[style*="pointer"]');
        expect(closeButtonGroup).toBeInTheDocument();

        closeButtonGroup?.dispatchEvent(new Event('pointerdown', { bubbles: true, cancelable: true }));

        expect(mockOnRemove).toHaveBeenCalledTimes(1);
        expect(mockOnRemove).toHaveBeenCalledWith('label-1');
    });

    it('adjusts sizes based on zoom scale', () => {
        mockZoom.scale = 2;
        const label = getMockedLabel({ name: 'Person' });

        const { container } = render(
            <svg>
                <AnnotationLabels labels={[label]} onRemove={mockOnRemove} />
            </svg>
        );

        const text = container.querySelector('text');
        const fontSize = text?.getAttribute('font-size');

        // Font size should be 14 / 2 = 7
        expect(fontSize).toBe('7');
    });

    it('prevents event propagation on close button click', () => {
        const label = getMockedLabel({ id: 'label-1' });
        const mockParentHandler = vi.fn();

        const { container } = render(
            <svg onPointerDown={mockParentHandler}>
                <AnnotationLabels labels={[label]} onRemove={mockOnRemove} />
            </svg>
        );

        const closeButtonGroup = container.querySelector('g[style*="pointer"]');
        const event = new Event('pointerdown', { bubbles: true, cancelable: true });

        closeButtonGroup?.dispatchEvent(event);

        expect(mockOnRemove).toHaveBeenCalled();
        // Parent handler should not be called due to stopPropagation
        expect(mockParentHandler).not.toHaveBeenCalled();
    });

    it('renders labels with correct positioning (no overlap)', () => {
        const labels: Label[] = [
            getMockedLabel({ id: '1', name: 'A', color: '#FF0000' }),
            getMockedLabel({ id: '2', name: 'B', color: '#00FF00' }),
        ];

        const { container } = render(
            <svg>
                <AnnotationLabels labels={labels} onRemove={mockOnRemove} />
            </svg>
        );

        const rects = container.querySelectorAll('rect');
        const firstX = parseFloat(rects[0]?.getAttribute('x') || '0');
        const secondX = parseFloat(rects[2]?.getAttribute('x') || '0');

        // Second label should be positioned after the first
        expect(secondX).toBeGreaterThan(firstX);
    });
});
