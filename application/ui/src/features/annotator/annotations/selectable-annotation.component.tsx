// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { KeyboardEvent, MouseEvent, ReactNode, useEffect, useRef } from 'react';

import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';
import { useSelectedAnnotations } from '../../../shared/annotator/select-annotation-provider.component';
import { drawingStyles } from '../tools/polygon-tool/utils';
import { useAnnotation } from './annotation-context';

export const SelectableAnnotation = ({ children }: { children: ReactNode }) => {
    const annotation = useAnnotation();
    const { deleteAnnotations } = useAnnotationActions();
    const { setSelectedAnnotations, selectedAnnotations } = useSelectedAnnotations();
    const elementRef = useRef<SVGGElement>(null);

    const isSelected = selectedAnnotations?.has(annotation.id);
    const selectionStyles = isSelected ? { stroke: 'var(--energy-blue-light)' } : {};

    // Focus the element when it becomes selected
    useEffect(() => {
        if (isSelected && elementRef.current) {
            elementRef.current.focus();
        }
    }, [isSelected]);

    const handleSelectAnnotation = (event: MouseEvent<SVGElement>) => {
        const hasShiftPressed = event.shiftKey;

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

            deleteAnnotations(annotationsToDelete);
        }
    };

    return (
        <g
            ref={elementRef}
            tabIndex={isSelected ? 0 : -1}
            onKeyDown={handleKeyDown}
            onClick={handleSelectAnnotation}
            style={{
                ...drawingStyles(annotation.labels[0]),
                ...selectionStyles,
                zIndex: 999,
                outline: 'none',
                position: 'relative',
                pointerEvents: 'auto',
            }}
        >
            {children}
        </g>
    );
};
