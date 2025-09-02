// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useId } from 'react-aria';

import { AnnotationShape } from '../../annotation-shape';
import { Annotation as AnnotationInterface } from '../../interfaces';

import classes from './segment-anything.module.scss';

const MASK_OPACITY = 0.4;

interface AnnotationMaskProps {
    annotations: AnnotationInterface[];
    width: number;
    height: number;
    fillOpacity?: number;
}

export const AnnotationsMask = ({ annotations, width, height, fillOpacity = MASK_OPACITY }: AnnotationMaskProps) => {
    const maskId = useId();

    return (
        <svg
            id={`annotations-mask`}
            aria-label='Annotation mask'
            width={width}
            height={height}
            className={classes.layer}
        >
            <mask id={maskId}>
                <rect x='0' y='0' width={width} height={height} fill='white' fillOpacity={fillOpacity} />
                {annotations.map((annotation) => (
                    <g key={annotation.id} fill='black'>
                        {/* Use an empty set of labels so that the shape won't receive a background */}
                        <AnnotationShape annotation={annotation} />
                    </g>
                ))}
            </mask>
            <rect x={0} y={0} width={width} height={height} mask={`url(#${maskId})`} color='black' />
        </svg>
    );
};
