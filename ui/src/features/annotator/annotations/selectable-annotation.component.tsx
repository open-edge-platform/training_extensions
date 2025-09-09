// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { MouseEvent, ReactNode } from 'react';

import { noop } from 'lodash-es';

import { useAnnotator } from '../annotator-provider.component';
import { useSelectedAnnotations, useSetSelectedAnnotations } from '../select-annotation-provider.component';
import { useAnnotation } from './annotation.component';

export const SelectableAnnotation = ({ children }: { children: ReactNode }) => {
    const annotation = useAnnotation();
    const isSelected = useSelectedAnnotations().has(annotation.id);
    const setSelectedAnnotations = useSetSelectedAnnotations();
    const { activeTool } = useAnnotator();

    const hasSelectionToolActive = activeTool === 'selection';

    const handleSelectAnnotation = (e: MouseEvent<SVGElement>) => {
        const hasShiftPressed = e.shiftKey;

        setSelectedAnnotations((selected) => {
            if (!hasShiftPressed) {
                return new Set([annotation.id]);
            }

            const newSelected = new Set(selected);

            if (newSelected.has(annotation.id)) {
                newSelected.delete(annotation.id);
            } else {
                newSelected.add(annotation.id);
            }

            return newSelected;
        });
    };

    return (
        <g
            onClick={hasSelectionToolActive ? handleSelectAnnotation : noop}
            style={{
                ...(isSelected
                    ? {
                          fillOpacity: 0.7,
                          ['--annotation-fill']: annotation.labels[0].color,
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
