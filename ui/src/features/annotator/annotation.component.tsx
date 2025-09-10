// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ShapeFactory } from '../../components/annotation/shape-factory.component';
import { Annotation as AnnotationInterface } from './types';

interface AnnotationProps {
    annotation: AnnotationInterface;
    maskId?: string;
}

export const Annotation = ({ maskId, annotation }: AnnotationProps) => {
    const { id, labels, shape } = annotation;

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
                <ShapeFactory shape={shape} styles={{}} ariaLabel={''} />
            </g>
        </>
    );
};
