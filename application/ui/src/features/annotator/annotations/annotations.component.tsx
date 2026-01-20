// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { MouseEvent } from 'react';

import { isEmpty } from 'lodash-es';

import { useSelectedAnnotations } from '../../../shared/annotator/select-annotation-provider.component';
import type { Annotation as AnnotationType } from '../../../shared/types';
import { DEFAULT_ANNOTATION_STYLES } from '../utils';
import { Annotation } from './annotation.component';
import { MaskAnnotations } from './mask-annotations.component';

type AnnotationsProps = {
    annotations: AnnotationType[];
    width: number;
    height: number;
    isFocussed: boolean;
};

export const Annotations = ({ annotations, width, height, isFocussed }: AnnotationsProps) => {
    const { setSelectedAnnotations } = useSelectedAnnotations();

    // If the user clicks on an empty spot on the canvas, we want to deselect
    // all annotations
    const handleBackgroundClick = (e: MouseEvent<SVGSVGElement>) => {
        if (e.target === e.currentTarget) {
            setSelectedAnnotations(new Set());
        }
    };

    return (
        <svg
            aria-label={'annotations'}
            width={width}
            height={height}
            tabIndex={-1}
            onClick={handleBackgroundClick}
            style={{
                position: 'absolute',
                inset: 0,
                outline: 'none',
                overflow: 'visible',
                ...DEFAULT_ANNOTATION_STYLES,
            }}
        >
            {!isEmpty(annotations) && (
                <MaskAnnotations annotations={annotations} width={width} height={height} isEnabled={isFocussed}>
                    {annotations.map((annotation) => (
                        <Annotation key={annotation.id} annotation={annotation} />
                    ))}
                </MaskAnnotations>
            )}
        </svg>
    );
};
