// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedLabel } from 'mocks/mock-labels';
import { fireEvent, render, screen } from 'test-utils/render';
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
        render(
            <svg>
                <AnnotationLabels labels={[]} onRemove={mockOnRemove} />
            </svg>
        );

        expect(screen.getByText('No label')).toBeInTheDocument();
    });

    it('renders single label with name and color', () => {
        const label = getMockedLabel({ name: 'Person', color: '#FF0000' });

        render(
            <svg>
                <AnnotationLabels labels={[label]} onRemove={mockOnRemove} />
            </svg>
        );

        expect(screen.getByText('Person')).toBeInTheDocument();

        const rect = screen.getByLabelText('label Person background');
        expect(rect).toHaveAttribute('fill', '#FF0000');
    });

    it('renders multiple labels horizontally', () => {
        const labels: Label[] = [
            getMockedLabel({ id: '1', name: 'Person', color: '#FF0000' }),
            getMockedLabel({ id: '2', name: 'Car', color: '#00FF00' }),
        ];

        render(
            <svg>
                <AnnotationLabels labels={labels} onRemove={mockOnRemove} />
            </svg>
        );

        expect(screen.getByText('Person')).toBeInTheDocument();
        expect(screen.getByText('Car')).toBeInTheDocument();
    });

    it('calls onRemove when close button clicked', () => {
        const label = getMockedLabel({ id: 'label-1', name: 'Person' });

        render(
            <svg>
                <AnnotationLabels labels={[label]} onRemove={mockOnRemove} />
            </svg>
        );

        const closeButton = screen.getByLabelText('Remove Person');
        fireEvent.pointerDown(closeButton);

        expect(mockOnRemove).toHaveBeenCalledTimes(1);
        expect(mockOnRemove).toHaveBeenCalledWith('label-1');
    });

    it('adjusts sizes based on zoom scale', () => {
        mockZoom.scale = 2;
        const label = getMockedLabel({ name: 'Person' });

        render(
            <svg>
                <AnnotationLabels labels={[label]} onRemove={mockOnRemove} />
            </svg>
        );

        const text = screen.getByLabelText('label Person');

        // Font size should be 14 / 2 = 7
        expect(text).toHaveAttribute('font-size', '7');
    });

    it('prevents event propagation on close button click', () => {
        const label = getMockedLabel({ id: 'label-1', name: 'Person' });
        const mockParentHandler = vi.fn();

        render(
            <svg onPointerDown={mockParentHandler}>
                <AnnotationLabels labels={[label]} onRemove={mockOnRemove} />
            </svg>
        );

        const closeButton = screen.getByLabelText('Remove Person');
        fireEvent.pointerDown(closeButton);

        expect(mockOnRemove).toHaveBeenCalled();
        // Parent handler should not be called due to stopPropagation
        expect(mockParentHandler).not.toHaveBeenCalled();
    });

    it('renders labels with correct positioning (no overlap)', () => {
        const labels: Label[] = [
            getMockedLabel({ id: '1', name: 'A', color: '#FF0000' }),
            getMockedLabel({ id: '2', name: 'B', color: '#00FF00' }),
        ];

        render(
            <svg>
                <AnnotationLabels labels={labels} onRemove={mockOnRemove} />
            </svg>
        );

        const firstRect = screen.getByLabelText('label A background');
        const secondRect = screen.getByLabelText('label B background');

        const firstX = parseFloat(firstRect.getAttribute('x') || '0');
        const secondX = parseFloat(secondRect.getAttribute('x') || '0');

        // Second label should be positioned after the first
        expect(secondX).toBeGreaterThan(firstX);
    });
});
