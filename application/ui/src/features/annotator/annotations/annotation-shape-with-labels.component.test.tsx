// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { getMockedAnnotation } from 'mocks/mock-annotation';
import { getMockedLabel } from 'mocks/mock-labels';
import { render } from 'test-utils/render';

import { AnnotationShapeWithLabels } from './annotation-shape-with-labels.component';

const mockDeleteAnnotations = vi.fn();
const mockUpdateAnnotations = vi.fn();

vi.mock('hooks/api/project.hook', () => ({
    useProject: () => ({
        data: {
            task: {
                task_type: 'detection',
            },
        },
    }),
}));

vi.mock('../../../shared/annotator/annotation-visibility-provider.component', () => ({
    useAnnotationVisibility: () => ({
        isVisible: true,
    }),
}));

vi.mock('../../../shared/annotator/annotation-actions-provider.component', () => ({
    useAnnotationActions: () => ({
        updateAnnotations: mockUpdateAnnotations,
        deleteAnnotations: mockDeleteAnnotations,
        isReadOnlyMode: false,
    }),
}));

vi.mock('../selected-media-item-provider.component', () => ({
    useSelectedMediaItem: () => ({
        mediaItem: {
            width: 100,
            height: 100,
        },
    }),
}));

describe('AnnotationShapeWithLabels', () => {
    beforeEach(() => {
        mockDeleteAnnotations.mockClear();
        mockUpdateAnnotations.mockClear();
    });

    it('deletes full-image annotation when removing its last label in detection projects', () => {
        const annotation = getMockedAnnotation({
            id: 'full-image-annotation',
            shape: { type: 'full_image' },
            labels: [getMockedLabel({ id: 'label-1', name: 'No object' })],
        });

        render(<AnnotationShapeWithLabels annotation={annotation} />);

        fireEvent.pointerDown(screen.getByRole('button', { name: 'Remove No object' }));

        expect(mockDeleteAnnotations).toHaveBeenCalledWith(['full-image-annotation']);
        expect(mockUpdateAnnotations).not.toHaveBeenCalled();
    });

    it('keeps non-full-image behavior unchanged in detection projects', () => {
        const annotation = getMockedAnnotation({
            id: 'rect-annotation',
            labels: [getMockedLabel({ id: 'label-1', name: 'Person' })],
        });

        render(<AnnotationShapeWithLabels annotation={annotation} />);

        fireEvent.pointerDown(screen.getByRole('button', { name: 'Remove Person' }));

        expect(mockDeleteAnnotations).not.toHaveBeenCalled();
        expect(mockUpdateAnnotations).toHaveBeenCalledWith([
            {
                ...annotation,
                labels: [],
            },
        ]);
    });
});
