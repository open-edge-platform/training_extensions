// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CSSProperties } from 'react';

import { AnnotationShape } from './annotation-shape';
import { MaskAnnotations } from './mask-annotations';
import { Annotation } from './types';

type AnnotationsProps = {
    annotations: Array<Annotation>;
    width: number;
    height: number;
    isFocussed: boolean;
};

const DEFAULT_ANNOTATION_STYLES = {
    fillOpacity: 0.4,
    fill: 'var(--annotation-fill)',
    stroke: 'var(--annotation-stroke)',
    strokeLinecap: 'round',
    strokeWidth: 'calc(1px / var(--zoom-scale))',
    strokeDashoffset: 0,
    strokeDasharray: 0,
    strokeOpacity: 'var(--annotation-border-opacity, 1)',
} satisfies CSSProperties;

export const Annotations = ({ annotations, width, height, isFocussed }: AnnotationsProps) => {
    return (
        <svg width={width} height={height} style={DEFAULT_ANNOTATION_STYLES}>
            <MaskAnnotations annotations={annotations} width={width} height={height} isEnabled={isFocussed}>
                {annotations.map((annotation) => (
                    <AnnotationShape key={annotation.id} annotation={annotation} />
                ))}
            </MaskAnnotations>
        </svg>
    );
};
