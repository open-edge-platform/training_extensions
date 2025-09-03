// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Annotation as AnnotationInterface } from '../shapes/interfaces';
import { ShapeFactory } from './shape-factory.component';

interface AnnotationProps {
    annotation: AnnotationInterface;
    maskId?: string;
}

export const Annotation = ({ maskId, annotation }: AnnotationProps) => {
    const { id, color, shape } = annotation;

    return (
        <>
            <g
                mask={maskId}
                id={`canvas-annotation-${id}`}
                strokeLinecap={'round'}
                {...(color !== undefined
                    ? {
                          fill: color,
                          stroke: color,
                          strokeOpacity: 'var(--annotation-border-opacity)',
                      }
                    : {})}
            >
                <ShapeFactory shape={shape} styles={{}} ariaLabel={''} />
            </g>
        </>
    );
};
