// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { KeyboardEvent, MouseEvent, ReactNode, useEffect, useRef } from 'react';

import { useAnnotationActions } from '../annotation-actions-provider.component';
import { useSelectedAnnotations } from '../select-annotation-provider.component';
import { useAnnotation } from './annotation-context';

export const SelectableAnnotation = ({ children }: { children: ReactNode }) => {
    const annotation = useAnnotation();
    const { deleteAnnotation } = useAnnotationActions();
    const { setSelectedAnnotations, selectedAnnotations } = useSelectedAnnotations();
    const elementRef = useRef<SVGGElement>(null);

    const isSelected = selectedAnnotations?.has(annotation.id);

    // Focus the element when it becomes selected
    useEffect(() => {
        if (isSelected && elementRef.current) {
            elementRef.current.focus();
        }
    }, [isSelected]);

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

    const handleKeyDown = (event: KeyboardEvent<SVGElement>) => {
        if (event.key === 'Backspace') {
            event.preventDefault();

            // Focus the parent SVG container to keep focus within the annotation area
            const parentSvg = elementRef.current?.closest('svg');
            if (parentSvg) {
                (parentSvg as SVGSVGElement).focus();
            }

            const annotationsToDelete = Array.from(selectedAnnotations);

            setSelectedAnnotations(new Set());

            annotationsToDelete.forEach((annotationId) => {
                deleteAnnotation(annotationId);
            });
        }
    };

    return (
        <g
            ref={elementRef}
            tabIndex={isSelected ? 0 : -1}
            onKeyDown={handleKeyDown}
            onClick={handleSelectAnnotation}
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
                position: 'relative',
                zIndex: 999,
                pointerEvents: 'auto',
                outline: 'none',
            }}
        >
            {children}
        </g>
    );
};
