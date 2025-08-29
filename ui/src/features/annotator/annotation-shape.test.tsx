// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { getMockedAnnotation } from '../../../tests/test-utils/mocked-annotation';
import { AnnotationShape } from './annotation-shape';
import { Annotation, BoundingBox, Circle, Polygon } from './types';

type AnnotationBoundingBox = Annotation & { shape: BoundingBox };
type AnnotationPolygon = Annotation & { shape: Polygon };
type AnnotationCircle = Annotation & { shape: Circle };

describe('AnnotationShape', () => {
    it('bounding box as a rect', () => {
        const annotation = getMockedAnnotation() as AnnotationBoundingBox;
        render(<AnnotationShape annotation={annotation} />);

        const rect = screen.getByLabelText('annotation bounding-box');

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

    it('circle for other shape types', () => {
        const annotation = getMockedAnnotation({
            shape: { type: 'circle', cx: 15, cy: 25, r: 30 },
        }) as AnnotationCircle;

        render(<AnnotationShape annotation={annotation} />);
        const circle = screen.getByLabelText('annotation circle');

        expect(circle).toHaveAttribute('cx', annotation.shape.cx.toString());
        expect(circle).toHaveAttribute('cy', annotation.shape.cy.toString());
        expect(circle).toHaveAttribute('r', annotation.shape.r.toString());
        expect(circle).toHaveAttribute('fill', annotation.labels[0].color);
    });
});
