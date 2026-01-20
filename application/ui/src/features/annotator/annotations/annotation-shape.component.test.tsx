// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedAnnotation } from 'mocks/mock-annotation';
import { render, screen } from 'test-utils/render';

import type { Annotation, Polygon, Rect } from '../../../shared/types';
import { AnnotationShape } from './annotation-shape.component';

type AnnotationRect = Annotation & { shape: Rect };
type AnnotationPolygon = Annotation & { shape: Polygon };

describe('AnnotationShape', () => {
    it('bounding box as a rect', () => {
        const annotation = getMockedAnnotation() as AnnotationRect;
        render(<AnnotationShape annotation={annotation} />);

        const rect = screen.getByLabelText('annotation rect');

        expect(rect).toHaveAttribute('x', annotation.shape.x.toString());
        expect(rect).toHaveAttribute('y', annotation.shape.y.toString());
        expect(rect).toHaveAttribute('fill', annotation.labels[0].color);
        expect(rect).toHaveAttribute('width', annotation.shape.width.toString());
        expect(rect).toHaveAttribute('height', annotation.shape.height.toString());
    });

    it('polygon', () => {
        const points = [
            { x: 1, y: 2 },
            { x: 3, y: 4 },
            { x: 5, y: 6 },
        ];
        const annotation = getMockedAnnotation({ shape: { type: 'polygon', points } }) as AnnotationPolygon;

        render(<AnnotationShape annotation={annotation} />);
        const polygon = screen.getByLabelText('annotation polygon');

        expect(polygon).toHaveAttribute('points', '1,2 3,4 5,6');
        expect(polygon).toHaveAttribute('fill', annotation.labels[0].color);
    });
});
