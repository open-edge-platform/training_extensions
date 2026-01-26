// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useId } from 'react';

import type { Annotation } from '../../../shared/types';
import { AnnotationShapeRenderer } from './annotation-shape-renderer.component';

type MaskAnnotationsProps = {
    annotations: Annotation[];
    children: ReactNode;
    width: number;
    height: number;
    isEnabled: boolean;
};

export const MaskAnnotations = ({ annotations, children, width, height, isEnabled }: MaskAnnotationsProps) => {
    const id = useId();
    const maskOpacity = isEnabled ? 0.8 : 0.0;

    return (
        <>
            <mask id={`mask-${id}`}>
                <rect x='0' y='0' width={width} height={height} style={{ fill: 'white', fillOpacity: 1.0 }} />
                {annotations.map((annotation, idx) => (
                    <g
                        key={idx}
                        style={{
                            fill: 'black',
                            fillOpacity: isEnabled ? 1.0 : 0.0,
                            transitionProperty: 'fill-opacity',
                            transitionTimingFunction: 'ease-in-out',
                            transitionDuration: isEnabled ? '0.2s' : '0.1s',
                            transitionDelay: isEnabled ? '0s' : '.25s',
                        }}
                    >
                        <AnnotationShapeRenderer annotation={annotation} />
                    </g>
                ))}
            </mask>
            <rect
                x={0}
                y={0}
                width={width}
                height={height}
                mask={`url(#mask-${id})`}
                pointerEvents={'none'}
                style={{
                    fillOpacity: maskOpacity,
                    fill: 'black',
                    strokeWidth: 0,
                    transition: 'fill-opacity 0.1s ease-in-out',
                    transitionDelay: isEnabled ? '0s' : '.25s',
                    transitionDuration: isEnabled ? '0.2s' : '0.1s',
                }}
            />
            {children}
        </>
    );
};
