// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Annotation } from '../../annotations/annotation.component';
import { useTranslate } from '../../hooks/use-translate.hook';
import { Annotation as AnnotationInterface } from '../../types';
import { allowPanning } from '../../utils';

const STROKE_WIDTH = 2;

interface TranslateShapeProps {
    zoom: number;
    annotation: AnnotationInterface;
    translateShape: ({ x, y }: { x: number; y: number }) => void;
    onComplete: () => void;
}

export const TranslateShape = ({ zoom, onComplete, translateShape, annotation }: TranslateShapeProps) => {
    const { onPointerDown, onPointerMove, onPointerUp } = useTranslate({
        zoom,
        onTranslate: translateShape,
        onComplete,
    });

    return (
        <g
            id={`translate-annotation-${annotation.id}`}
            stroke='var(--energy-blue)'
            strokeWidth={STROKE_WIDTH / zoom}
            aria-label='Drag to move shape'
            style={{ pointerEvents: 'auto', cursor: 'move' }}
            onPointerDown={allowPanning(onPointerDown)}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerLeave={onPointerUp}
        >
            <Annotation annotation={annotation} />
        </g>
    );
};
