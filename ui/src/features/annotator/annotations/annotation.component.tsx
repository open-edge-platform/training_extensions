// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Annotation as AnnotationInterface } from '../types';
import { AnnotationShape } from './annotation-shape';

interface AnnotationProps {
    annotation: AnnotationInterface;
    maskId?: string;
}

export const Annotation = ({ maskId, annotation }: AnnotationProps) => {
    const { id, labels } = annotation;

    return (
        <>
            <g
                mask={maskId}
                id={`canvas-annotation-${id}`}
                strokeLinecap={'round'}
                {...(labels.length > 0
                    ? {
                          fill: labels[0].color,
                          stroke: labels[0].color,
                          strokeOpacity: 'var(--annotation-border-opacity)',
                      }
                    : {})}
            >
                <AnnotationShape annotation={annotation} />
            </g>
        </>
    );
};
