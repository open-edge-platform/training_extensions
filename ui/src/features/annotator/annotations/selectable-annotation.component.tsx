// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { useAnnotator } from '../annotator-provider.component';
import { useAnnotation } from './annotation.component';

type SelectableAnnotationProps = {
    children: ReactNode;
};

export const SelectableAnnotation = ({ children }: SelectableAnnotationProps) => {
    const { setSelectedAnnotation, selectedAnnotation } = useAnnotator();
    const annotation = useAnnotation();

    const isSelected = selectedAnnotation?.id === annotation.id;

    return (
        <g
            onClick={() => setSelectedAnnotation(annotation)}
            style={{
                ...(isSelected
                    ? {
                          fillOpacity: 0.7,
                          cursor: 'move',
                          stroke: 'var(--energy-blue-light)',
                          strokeWidth: 'calc(2px / var(--zoom-scale))',
                      }
                    : {}),
                transitionProperty: 'fill-opacity',
                transitionTimingFunction: 'ease-in-out',
                transitionDuration: '0.1s',
                transitionDelay: isSelected ? '0s' : '0.25s',
            }}
        >
            {children}
        </g>
    );
};
