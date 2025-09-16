// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, useEffect, useRef, useState } from 'react';

import { clampBox, clampPointBetweenImage, pointsToRect } from '@geti/smart-tools/utils';
import { type KeyboardEvent as ReactKeyboardEvent } from '@geti/ui';
import { isFunction } from 'lodash-es';

import { useAnnotator } from '../../annotator-provider.component';
import { Rectangle } from '../../shapes/rectangle.component';
import { Point, Rect as RectInterface } from '../../types';
import { isEraserButton, isLeftButton } from '../../utils';
import { Crosshair } from './crosshair/crosshair.component';
import { getRelativePoint, useCrosshair } from './crosshair/utils';

enum PointerType {
    Mouse = 'mouse',
    Pen = 'pen',
    Touch = 'touch',
}

const CURSOR_OFFSET = '7 8';
interface DrawingBoxInterface {
    zoom: number;
    image: ImageData;
    withCrosshair?: boolean;
    onStart?: () => void;
    onComplete: (shapes: RectInterface) => void;
}

export const DrawingBox = ({ image, zoom, onStart, onComplete, withCrosshair = true }: DrawingBoxInterface) => {
    const { mediaItem } = useAnnotator();
    const roi = { x: 0, y: 0, width: mediaItem.width, height: mediaItem.height };

    const [startPoint, setStartPoint] = useState<Point | null>(null);
    const [boundingBox, setBoundingBox] = useState<RectInterface | null>(null);
    const [hasCrossHair, setHasCrossHair] = useState<boolean>(withCrosshair);
    const ref = useRef<SVGRectElement>(null);
    const clampPoint = clampPointBetweenImage(image);
    const crosshair = useCrosshair(ref, zoom);

    const onPointerMove = (event: PointerEvent<SVGSVGElement>): void => {
        crosshair.onPointerMove(event);

        if (ref.current === null) {
            return;
        }

        const button = {
            button: event.button,
            buttons: event.buttons,
        };

        if (event.pointerType === PointerType.Pen && isEraserButton(button)) {
            setHasCrossHair(false);

            return;
        } else {
            setHasCrossHair(withCrosshair);
        }

        if (startPoint === null || !event.currentTarget.hasPointerCapture(event.pointerId)) {
            return;
        }

        const endPoint = clampPoint(getRelativePoint(ref.current, { x: event.clientX, y: event.clientY }, zoom));

        setBoundingBox({ shapeType: 'rect', ...clampBox(pointsToRect(startPoint, endPoint), roi) });
    };

    const onPointerDown = (event: PointerEvent<SVGSVGElement>): void => {
        if (startPoint !== null || ref.current === null) {
            return;
        }

        const button = {
            button: event.button,
            buttons: event.buttons,
        };

        if (event.pointerType === PointerType.Touch || !isLeftButton(button)) {
            return;
        }

        isFunction(onStart) && onStart();

        const mouse = clampPoint(getRelativePoint(ref.current, { x: event.clientX, y: event.clientY }, zoom));

        event.currentTarget.setPointerCapture(event.pointerId);

        setStartPoint(mouse);
        setBoundingBox({ shapeType: 'rect', x: mouse.x, y: mouse.y, width: 0, height: 0 });
    };

    const onPointerUp = (event: PointerEvent<SVGSVGElement>): void => {
        if (event.pointerType === PointerType.Touch) {
            return;
        }

        if (boundingBox === null) {
            return;
        }

        // Don't make empty annotations
        if (boundingBox.width > 1 && boundingBox.height > 1) {
            onComplete(boundingBox);
        }

        setCleanState();

        event.currentTarget.releasePointerCapture(event.pointerId);
    };

    const setCleanState = () => {
        setStartPoint(null);
        setBoundingBox(null);
    };

    useEffect(() => {
        window.addEventListener('keydown', ({ key }: KeyboardEvent | ReactKeyboardEvent) => {
            if (key === 'Escape') {
                setCleanState();
            }
        });

        return () => {
            window.removeEventListener('keydown', () => null);
        };
    }, []);

    return (
        <svg
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerDown={onPointerDown}
            // eslint-disable-next-line jsx-a11y/aria-role
            role='editor'
            viewBox={`0 0 ${roi.width} ${roi.height}`}
            style={{
                cursor: withCrosshair ? `url(/icons/cursor/selection.svg) ${CURSOR_OFFSET}, auto` : 'default',
            }}
        >
            <rect {...roi} fillOpacity={0} ref={ref} />
            {boundingBox ? (
                <Rectangle ariaLabel={'bounding box'} rect={boundingBox} styles={{ role: 'application' }} />
            ) : null}
            {hasCrossHair && <Crosshair location={crosshair.location} zoom={zoom} />}
        </svg>
    );
};
