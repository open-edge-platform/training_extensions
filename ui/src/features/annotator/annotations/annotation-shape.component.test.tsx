// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { getMockedAnnotation } from '../../../../tests/test-utils/mocked-annotation';
import { response } from '../../dataset/mock-response';
import { AnnotatorProvider } from '../annotator-provider.component';
import { Annotation, Polygon, Rect } from '../types';
import { AnnotationShape } from './annotation-shape.component';

type AnnotationRect = Annotation & { shape: Rect };
type AnnotationPolygon = Annotation & { shape: Polygon };

const mockMediaItem = response.items[0];

describe('AnnotationShape', () => {
    it('bounding box as a rect', () => {
        const annotation = getMockedAnnotation() as AnnotationRect;
        render(
            <AnnotatorProvider mediaItem={mockMediaItem}>
                <AnnotationShape annotation={annotation} />
            </AnnotatorProvider>
        );

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
        const annotation = getMockedAnnotation({ shape: { shapeType: 'polygon', points } }) as AnnotationPolygon;

        render(
            <AnnotatorProvider mediaItem={mockMediaItem}>
                <AnnotationShape annotation={annotation} />
            </AnnotatorProvider>
        );
        const polygon = screen.getByLabelText('annotation polygon');

        expect(polygon).toHaveAttribute('points', '1,2 3,4 5,6');
        expect(polygon).toHaveAttribute('fill', annotation.labels[0].color);
    });
});
