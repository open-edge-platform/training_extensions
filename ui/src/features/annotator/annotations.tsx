// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CSSProperties } from 'react';

import { AnnotationShape } from './annotation-shape';
import { useAnnotator } from './annotator-provider.component';
import { MaskAnnotations } from './mask-annotations';

type AnnotationsProps = {
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

export const Annotations = ({ width, height, isFocussed }: AnnotationsProps) => {
    const { annotations, selectedAnnotation } = useAnnotator();

    // Hide the ones being edited (resized or translated)
    const staticAnnotations = annotations.filter((a) => a.id !== selectedAnnotation?.id);

    return (
        <svg width={width} height={height} style={DEFAULT_ANNOTATION_STYLES}>
            <MaskAnnotations width={width} height={height} isEnabled={isFocussed}>
                {staticAnnotations.map((annotation) => (
                    <AnnotationShape key={annotation.id} annotation={annotation} />
                ))}
            </MaskAnnotations>
        </svg>
    );
};
