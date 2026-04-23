// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { getMockedAnnotation } from 'mocks/mock-annotation';
import { getMockedLabel } from 'mocks/mock-labels';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { EMPTY_LABEL_ID } from '../../../shared/annotator/labels';
import { AnnotationShapeWithLabels } from './annotation-shape-with-labels.component';

const mockDeleteAnnotations = vi.fn();
const mockUpdateAnnotations = vi.fn();
const mockSetSelectedLabelId = vi.fn();

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

vi.mock('../annotator-labels-provider.component', () => ({
    useAnnotatorLabels: () => ({
        labels: [],
        selectedLabel: null,
        selectedLabelId: 'empty-label',
        setSelectedLabelId: mockSetSelectedLabelId,
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
        const project = getMockedProject({});

        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json({
                    ...project,
                    task: {
                        ...project.task,
                        task_type: 'detection',
                    },
                });
            })
        );

        vi.clearAllMocks();
    });

    it('deletes full-image annotation when removing its last label in detection projects', async () => {
        const annotation = getMockedAnnotation({
            id: 'full-image-annotation',
            shape: { type: 'full_image' },
            labels: [getMockedLabel({ id: EMPTY_LABEL_ID, name: 'No object' })],
        });

        render(<AnnotationShapeWithLabels annotation={annotation} />);
        fireEvent.pointerDown(await screen.findByRole('button', { name: 'Remove No object' }));

        expect(mockDeleteAnnotations).toHaveBeenCalledWith(['full-image-annotation']);
        expect(mockUpdateAnnotations).not.toHaveBeenCalled();
        expect(mockSetSelectedLabelId).toHaveBeenCalledWith(null);
    });

    it('deletes non-full-image annotation when removing its last label in detection projects', async () => {
        const annotation = getMockedAnnotation({
            id: 'rect-annotation',
            labels: [getMockedLabel({ id: 'label-1', name: 'Person' })],
        });

        render(<AnnotationShapeWithLabels annotation={annotation} />);
        fireEvent.pointerDown(await screen.findByRole('button', { name: 'Remove Person' }));

        expect(mockDeleteAnnotations).toHaveBeenCalledWith(['rect-annotation']);
        expect(mockUpdateAnnotations).not.toHaveBeenCalled();
    });
});
