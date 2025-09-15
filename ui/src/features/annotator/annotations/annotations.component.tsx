// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CSSProperties, MouseEvent } from 'react';

import { useAnnotator } from '../annotator-provider.component';
import { useSelectedAnnotations } from '../select-annotation-provider.component';
import { Annotation } from './annotation.component';
import { MaskAnnotations } from './mask-annotations.component';

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
    const { annotations } = useAnnotator();
    const { setSelectedAnnotations, selectedAnnotations } = useSelectedAnnotations();

    // Order annotations by selection. Selected annotation should always be on top.
    const orderedAnnotations = [
        ...annotations.filter((a) => !selectedAnnotations.has(a.id)),
        ...annotations.filter((a) => selectedAnnotations.has(a.id)),
    ];

    const handleClickOutside = (e: MouseEvent<SVGSVGElement>): void => {
        if (e.target === e.currentTarget) {
            setSelectedAnnotations(new Set());
        }
    };

    return (
        <svg width={width} height={height} style={DEFAULT_ANNOTATION_STYLES} onClick={handleClickOutside}>
            <MaskAnnotations annotations={orderedAnnotations} width={width} height={height} isEnabled={isFocussed}>
                {orderedAnnotations.map((annotation) => (
                    <Annotation annotation={annotation} key={annotation.id} />
                ))}
            </MaskAnnotations>
        </svg>
    );
};
