// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, useState } from 'react';

import { Annotation } from '../../annotation.component';
import { Annotation as AnnotationInterface, Point } from '../../types';
import { allowPanning, isLeftButton } from '../../utils';

const STROKE_WIDTH = 2;

interface TranslateShapeProps {
    zoom: number;
    annotation: AnnotationInterface;
    translateShape: ({ x, y }: { x: number; y: number }) => void;
    onComplete: () => void;
}

export const TranslateShape = ({ zoom, annotation, onComplete, translateShape }: TranslateShapeProps) => {
    const [dragFromPoint, setDragFromPoint] = useState<null | Point>(null);

    const onPointerDown = (event: PointerEvent<SVGSVGElement>): void => {
        if (dragFromPoint !== null) {
            return;
        }

        if (event.pointerType === 'touch' || !isLeftButton(event)) {
            return;
        }

        const mouse = { x: Math.round(event.clientX / zoom), y: Math.round(event.clientY / zoom) };

        event.currentTarget.setPointerCapture(event.pointerId);

        setDragFromPoint(mouse);
    };

    const onPointerMove = (event: PointerEvent<SVGSVGElement>) => {
        event.preventDefault();

        if (dragFromPoint === null) {
            return;
        }

        const mouse = { x: Math.round(event.clientX / zoom), y: Math.round(event.clientY / zoom) };

        translateShape({
            x: mouse.x - dragFromPoint.x,
            y: mouse.y - dragFromPoint.y,
        });

        setDragFromPoint(mouse);
    };

    const onPointerUp = (event: PointerEvent<SVGSVGElement>) => {
        if (dragFromPoint === null) {
            return;
        }

        event.preventDefault();
        setDragFromPoint(null);
        event.currentTarget.releasePointerCapture(event.pointerId);
        onComplete();
    };

    return (
        <g
            id={`translate-annotation-${annotation.id}`}
            stroke='var(--energy-blue)'
            strokeWidth={STROKE_WIDTH / zoom}
            aria-label='Drag to move shape'
            onPointerDown={allowPanning(onPointerDown)}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerLeave={onPointerUp}
            style={{ pointerEvents: 'auto', cursor: 'move' }}
        >
            <Annotation annotation={annotation} />
        </g>
    );
};
