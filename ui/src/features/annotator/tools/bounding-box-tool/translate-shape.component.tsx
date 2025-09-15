// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { useTranslate } from '../../hooks/use-translate.hook';
import { Annotation as AnnotationInterface } from '../../types';
import { allowPanning } from '../../utils';

const STROKE_WIDTH = 2;

interface TranslateShapeProps {
    zoom: number;
    annotation: AnnotationInterface;
    translateShape: ({ x, y }: { x: number; y: number }) => void;
    onComplete: () => void;
    children: ReactNode;
}

export const TranslateShape = ({ zoom, onComplete, translateShape, annotation, children }: TranslateShapeProps) => {
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
            onPointerDown={allowPanning(onPointerDown)}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerLeave={onPointerUp}
            style={{ cursor: 'move' }}
        >
            {children}
        </g>
    );
};
