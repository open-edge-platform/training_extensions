// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { getMockedLabel } from 'mocks/mock-labels';
import { render } from 'test-utils/render';

import { ZoomState } from '../../../../components/zoom/types';
import { useZoom } from '../../../../components/zoom/zoom.provider';
import type { Label } from '../../../../constants/shared-types';
import { AnnotationLabels } from './annotation-labels.component';

const mockZoom = { scale: 1, maxZoomIn: 10, hasAnimation: false, translate: { x: 0, y: 0 } };

vi.mock('../../../../components/zoom/zoom.provider', () => ({
    useZoom: vi.fn(() => mockZoom),
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
        const label = getMockedLabel({ name: 'Person' });
        vi.mocked(useZoom).mockReturnValue({ ...mockZoom, scale: 2 } as ZoomState);

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

        const firstPath = screen.getByLabelText('label A background');
        const secondPath = screen.getByLabelText('label B background');

        // Extract x position from path's d attribute (format: "M x y ...")
        const getPathX = (path: HTMLElement) => {
            const d = path.getAttribute('d') || '';
            const match = d.match(/^M\s+([-\d.]+)/);
            return match ? parseFloat(match[1]) : 0;
        };

        const firstX = getPathX(firstPath);
        const secondX = getPathX(secondPath);

        // Second label should be positioned after the first
        expect(secondX).toBeGreaterThan(firstX);
    });
});
