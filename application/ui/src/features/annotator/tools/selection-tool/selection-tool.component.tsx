// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, useRef, useState } from 'react';

import { clampBox, clampPointBetweenImage, pointsToRect } from '@geti/smart-tools/utils';
import { useEventListener } from 'hooks/event-listener.hook';

import { useZoom } from '../../../../components/zoom/zoom.provider';
import { useAnnotationActions } from '../../../../shared/annotator/annotation-actions-provider.component';
import { useSelectedAnnotations } from '../../../../shared/annotator/select-annotation-provider.component';
import { isLeftButton } from '../../../../shared/buttons-utils';
import type { Point, Rect, Rect as RectInterface } from '../../../../shared/types';
import { useSelectedMediaItem } from '../../selected-media-item-provider.component';
import { Rectangle } from '../../shapes/rectangle.component';
import { DEFAULT_ANNOTATION_STYLES } from '../../utils';
import { SvgToolCanvas } from '../svg-tool-canvas.component';
import { getRelativePoint, PointerType } from '../utils';
import { useClickWithoutDragging } from './use-click-without-dragging.hook';
import { getIntersectedAnnotationsIds, getTheTopShapeAt } from './utils';

export const SelectionTool = () => {
    const { scale: zoom } = useZoom();
    const { roi, image } = useSelectedMediaItem();
    const { annotations } = useAnnotationActions();
    const { setSelectedAnnotations } = useSelectedAnnotations();

    const [startPoint, setStartPoint] = useState<Point | null>(null);
    const [selectionBox, setSelectionBox] = useState<RectInterface | null>(null);

    const ref = useRef<SVGRectElement>(null);
    const selectingContainerRef = useRef<SVGSVGElement>(null);

    const clampPoint = clampPointBetweenImage(image);

    const onPointerDown = (event: PointerEvent<SVGSVGElement>): void => {
        if (startPoint !== null || ref.current === null) {
            return;
        }

        if (
            event.pointerType === PointerType.Touch ||
            !isLeftButton({ button: event.button, buttons: event.buttons })
        ) {
            return;
        }

        const mouse = clampPoint(getRelativePoint(ref.current, { x: event.clientX, y: event.clientY }, zoom));

        event.currentTarget.setPointerCapture(event.pointerId);

        setStartPoint(mouse);
        setSelectionBox({ type: 'rectangle', x: mouse.x, y: mouse.y, width: 0, height: 0 });
    };

    const onPointerMove = (event: PointerEvent<SVGSVGElement>): void => {
        if (ref.current === null || startPoint === null) {
            return;
        }

        if (!event.currentTarget.hasPointerCapture(event.pointerId)) {
            return;
        }

        const endPoint = clampPoint(getRelativePoint(ref.current, { x: event.clientX, y: event.clientY }, zoom));

        const newSelectionBox: Rect = { type: 'rectangle', ...clampBox(pointsToRect(startPoint, endPoint), roi) };
        const hightLightAnnotations = getIntersectedAnnotationsIds(annotations, newSelectionBox);

        setSelectionBox(newSelectionBox);
        setSelectedAnnotations(new Set(hightLightAnnotations));
    };

    const onPointerUp = (event: PointerEvent<SVGSVGElement>): void => {
        if (event.pointerType === PointerType.Touch) {
            return;
        }

        setStartPoint(null);
        setSelectionBox(null);

        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
        }
    };

    useEventListener('keydown', (event: KeyboardEvent) => {
        if (event.key === 'Escape') {
            setStartPoint(null);
            setSelectionBox(null);
        }
    });

    const handleClick = (event: globalThis.PointerEvent): void => {
        event.preventDefault();

        if (selectingContainerRef.current === null) {
            return;
        }

        const clickPoint = { x: event.clientX, y: event.clientY };
        const calculatePoint = getRelativePoint(selectingContainerRef.current, clickPoint, zoom);
        const points = clampPointBetweenImage(image)(calculatePoint);
        const highlightedAnnotation = getTheTopShapeAt(annotations, points);

        if (!highlightedAnnotation) {
            setSelectedAnnotations(new Set());
            return;
        }

        setSelectedAnnotations((selected) => {
            if (!event.shiftKey) {
                return new Set([highlightedAnnotation.id]);
            }

            const newSelected = new Set(selected);

            if (newSelected.has(highlightedAnnotation.id)) {
                newSelected.delete(highlightedAnnotation.id);
            } else {
                newSelected.add(highlightedAnnotation.id);
            }

            return newSelected;
        });
    };

    useClickWithoutDragging(selectingContainerRef, handleClick);

    return (
        <SvgToolCanvas
            image={image}
            ref={selectingContainerRef}
            canvasRef={ref}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
        >
            {selectionBox ? (
                <Rectangle ariaLabel={'selection box'} rect={selectionBox} styles={DEFAULT_ANNOTATION_STYLES} />
            ) : null}
        </SvgToolCanvas>
    );
};
