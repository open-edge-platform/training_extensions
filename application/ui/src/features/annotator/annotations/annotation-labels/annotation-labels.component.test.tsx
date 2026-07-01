// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { getMockedLabel } from 'mocks/mock-labels';
import { render } from 'test-utils/render';

import type { Label } from '../../../../constants/shared-types';
import { AnnotationLabels } from './annotation-labels.component';

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

        const labelElement = screen.getByLabelText('label Person background');
        expect(labelElement).toHaveStyle({ '--label-color': '#FF0000' });
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

    it('does not render remove button when labels are non-removable', () => {
        const label = getMockedLabel({ id: 'label-1', name: 'Person' });

        render(
            <svg>
                <AnnotationLabels labels={[label]} onRemove={mockOnRemove} isRemovable={false} />
            </svg>
        );

        expect(screen.queryByLabelText('Remove Person')).not.toBeInTheDocument();
    });

    it('positions foreignObject just above annotation anchor for CSS scaling', () => {
        const label = getMockedLabel({ name: 'Person' });

        render(
            <svg>
                <AnnotationLabels labels={[label]} onRemove={mockOnRemove} />
            </svg>
        );

        const foreignObject = document.querySelector('foreignObject');
        expect(foreignObject).toHaveAttribute('height', '25');
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

    it('renders labels in correct order', () => {
        const labels: Label[] = [
            getMockedLabel({ id: '1', name: 'First', color: '#FF0000' }),
            getMockedLabel({ id: '2', name: 'Second', color: '#00FF00' }),
        ];

        render(
            <svg>
                <AnnotationLabels labels={labels} onRemove={mockOnRemove} />
            </svg>
        );

        const firstLabel = screen.getByLabelText('label First background');
        const secondLabel = screen.getByLabelText('label Second background');

        expect(firstLabel).toBeInTheDocument();
        expect(secondLabel).toBeInTheDocument();

        expect(firstLabel.parentElement).toBe(secondLabel.parentElement);
    });

    it('applies correct border radius for first and last labels', () => {
        const labels: Label[] = [
            getMockedLabel({ id: '1', name: 'First', color: '#FF0000' }),
            getMockedLabel({ id: '2', name: 'Middle', color: '#00FF00' }),
            getMockedLabel({ id: '3', name: 'Last', color: '#0000FF' }),
        ];

        render(
            <svg>
                <AnnotationLabels labels={labels} onRemove={mockOnRemove} />
            </svg>
        );

        const firstLabel = screen.getByLabelText('label First background');
        const middleLabel = screen.getByLabelText('label Middle background');
        const lastLabel = screen.getByLabelText('label Last background');

        // First label should have top-left rounded
        expect(firstLabel).toHaveStyle({
            '--border-top-left': 'var(--spectrum-global-dimension-size-50)',
            '--border-top-right': '0',
        });
        // Middle label should have no rounded corners
        expect(middleLabel).toHaveStyle({ '--border-top-left': '0', '--border-top-right': '0' });
        // Last label should have top-right rounded
        expect(lastLabel).toHaveStyle({
            '--border-top-left': '0',
            '--border-top-right': 'var(--spectrum-global-dimension-size-50)',
        });
    });

    it('applies bottom corners when useBottomCorners is true', () => {
        const labels: Label[] = [
            getMockedLabel({ id: '1', name: 'First', color: '#FF0000' }),
            getMockedLabel({ id: '2', name: 'Last', color: '#00FF00' }),
        ];

        render(
            <svg>
                <AnnotationLabels labels={labels} onRemove={mockOnRemove} useBottomCorners />
            </svg>
        );

        const firstLabel = screen.getByLabelText('label First background');
        const lastLabel = screen.getByLabelText('label Last background');

        // First label should have bottom-left rounded
        expect(firstLabel).toHaveStyle({
            '--border-bottom-left': 'var(--spectrum-global-dimension-size-50)',
            '--border-bottom-right': '0',
        });
        // Last label should have bottom-right rounded
        expect(lastLabel).toHaveStyle({
            '--border-bottom-left': '0',
            '--border-bottom-right': 'var(--spectrum-global-dimension-size-50)',
        });
    });
});
