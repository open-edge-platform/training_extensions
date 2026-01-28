// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { getMockedLabel } from 'mocks/mock-labels';
import { render } from 'test-utils/render';

import type { Label } from '../../../constants/shared-types';
import { Labels } from './labels.component';

const mockLabels: Label[] = [
    getMockedLabel({ id: 'label-1', name: 'Person', color: '#FF0000' }),
    getMockedLabel({ id: 'label-2', name: 'Car', color: '#00FF00' }),
    getMockedLabel({ id: 'label-3', name: 'Dog', color: '#0000FF' }),
];

const mockSetSelectedLabelId = vi.fn();
const mockUpdateAnnotations = vi.fn();
const mockSelectedAnnotations = { current: new Set<string>() };
const mockAnnotations = { current: [] as { id: string; labels: Label[]; shape: { type: string } }[] };

vi.mock('../../../shared/annotator/annotator-provider.component', () => ({
    useAnnotator: () => ({
        labels: mockLabels,
        selectedLabelId: 'label-1',
        setSelectedLabelId: mockSetSelectedLabelId,
    }),
}));

vi.mock('../../../shared/annotator/select-annotation-provider.component', () => ({
    useSelectedAnnotations: () => ({
        selectedAnnotations: mockSelectedAnnotations.current,
    }),
}));

vi.mock('../../../shared/annotator/annotation-actions-provider.component', () => ({
    useAnnotationActions: () => ({
        annotations: mockAnnotations.current,
        updateAnnotations: mockUpdateAnnotations,
    }),
}));

describe('Labels', () => {
    beforeEach(() => {
        mockSetSelectedLabelId.mockClear();
        mockUpdateAnnotations.mockClear();
        mockSelectedAnnotations.current = new Set();
        mockAnnotations.current = [];
    });

    it('renders all labels as badges', () => {
        render(<Labels />);

        expect(screen.getByRole('button', { name: 'Label Person' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Label Car' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Label Dog' })).toBeInTheDocument();
    });

    it('shows selected label with aria-pressed true', () => {
        render(<Labels />);

        const personButton = screen.getByRole('button', { name: 'Label Person' });
        const carButton = screen.getByRole('button', { name: 'Label Car' });

        expect(personButton).toHaveAttribute('aria-pressed', 'true');
        expect(carButton).toHaveAttribute('aria-pressed', 'false');
    });

    it('calls setSelectedLabelId when clicking a label', () => {
        render(<Labels />);

        const carButton = screen.getByRole('button', { name: 'Label Car' });
        fireEvent.click(carButton);

        expect(mockSetSelectedLabelId).toHaveBeenCalledWith('label-2');
    });

    it('displays label names in badges', () => {
        render(<Labels />);

        expect(screen.getByText('Person')).toBeInTheDocument();
        expect(screen.getByText('Car')).toBeInTheDocument();
        expect(screen.getByText('Dog')).toBeInTheDocument();
    });

    it('updates selected annotations when clicking a different label', () => {
        mockSelectedAnnotations.current = new Set(['annotation-1']);
        mockAnnotations.current = [
            { id: 'annotation-1', labels: [mockLabels[0]], shape: { type: 'RECTANGLE' } },
            { id: 'annotation-2', labels: [mockLabels[1]], shape: { type: 'RECTANGLE' } },
        ];

        render(<Labels />);

        const carButton = screen.getByRole('button', { name: 'Label Car' });
        fireEvent.click(carButton);

        expect(mockSetSelectedLabelId).toHaveBeenCalledWith('label-2');
        expect(mockUpdateAnnotations).toHaveBeenCalledWith([mockAnnotations.current[0]], [mockLabels[1]]);
    });

    it('does not update annotations when no annotations are selected', () => {
        mockSelectedAnnotations.current = new Set();
        mockAnnotations.current = [{ id: 'annotation-1', labels: [mockLabels[0]], shape: { type: 'RECTANGLE' } }];

        render(<Labels />);

        const carButton = screen.getByRole('button', { name: 'Label Car' });
        fireEvent.click(carButton);

        expect(mockSetSelectedLabelId).toHaveBeenCalledWith('label-2');
        expect(mockUpdateAnnotations).not.toHaveBeenCalled();
    });

    it('removes label when all selected annotations already have it', () => {
        mockSelectedAnnotations.current = new Set(['annotation-1', 'annotation-2']);
        mockAnnotations.current = [
            { id: 'annotation-1', labels: [mockLabels[0]], shape: { type: 'RECTANGLE' } },
            { id: 'annotation-2', labels: [mockLabels[0]], shape: { type: 'RECTANGLE' } },
        ];

        render(<Labels />);

        const personButton = screen.getByRole('button', { name: 'Label Person' });
        fireEvent.click(personButton);

        expect(mockUpdateAnnotations).toHaveBeenCalledWith([
            { id: 'annotation-1', labels: [], shape: { type: 'RECTANGLE' } },
            { id: 'annotation-2', labels: [], shape: { type: 'RECTANGLE' } },
        ]);
        expect(mockSetSelectedLabelId).toHaveBeenCalledWith(null);
    });

    it('adds label when at least one selected annotation does not have it', () => {
        mockSelectedAnnotations.current = new Set(['annotation-1', 'annotation-2']);
        mockAnnotations.current = [
            { id: 'annotation-1', labels: [mockLabels[0]], shape: { type: 'RECTANGLE' } },
            { id: 'annotation-2', labels: [mockLabels[1]], shape: { type: 'RECTANGLE' } },
        ];

        render(<Labels />);

        const personButton = screen.getByRole('button', { name: 'Label Person' });
        fireEvent.click(personButton);

        expect(mockSetSelectedLabelId).toHaveBeenCalledWith('label-1');
        expect(mockUpdateAnnotations).toHaveBeenCalledWith(
            [mockAnnotations.current[0], mockAnnotations.current[1]],
            [mockLabels[0]]
        );
    });
});
