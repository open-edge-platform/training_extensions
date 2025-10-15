// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, useEffect, useRef, useState } from 'react';

import { clampBox, clampPointBetweenImage, pointsToRect } from '@geti/smart-tools/utils';
import { type KeyboardEvent as ReactKeyboardEvent } from '@geti/ui';

import selectionCursor from '../../../../assets/icons/selection.svg?url';
import { Rectangle } from '../../shapes/rectangle.component';
import { Point, Rect as RectInterface, RegionOfInterest } from '../../types';
import { DEFAULT_ANNOTATION_STYLES, isLeftButton } from '../../utils';
import { SvgToolCanvas } from '../svg-tool-canvas.component';
import { getRelativePoint } from '../utils';
import { Crosshair } from './crosshair/crosshair.component';
import { useCrosshair } from './crosshair/use-crosshair.hook';

enum PointerType {
    Mouse = 'mouse',
    Pen = 'pen',
    Touch = 'touch',
}

const CURSOR_OFFSET = '7 8';
interface DrawingBoxInterface {
    onComplete: (shapes: RectInterface) => void;
    roi: RegionOfInterest;
    image: ImageData;
    zoom: number;
}

export const DrawingBox = ({ roi, zoom, image, onComplete }: DrawingBoxInterface) => {
    const [startPoint, setStartPoint] = useState<Point | null>(null);
    const [boundingBox, setBoundingBox] = useState<RectInterface | null>(null);

    const ref = useRef<SVGRectElement>(null);

    const clampPoint = clampPointBetweenImage(image);
    const crosshair = useCrosshair(ref, zoom);

    const onPointerMove = (event: PointerEvent<SVGSVGElement>): void => {
        crosshair.onPointerMove(event);

        if (ref.current === null) {
            return;
        }

        if (startPoint === null || !event.currentTarget.hasPointerCapture(event.pointerId)) {
            return;
        }

        const endPoint = clampPoint(getRelativePoint(ref.current, { x: event.clientX, y: event.clientY }, zoom));

        setBoundingBox({ type: 'rectangle', ...clampBox(pointsToRect(startPoint, endPoint), roi) });
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

        const mouse = clampPoint(getRelativePoint(ref.current, { x: event.clientX, y: event.clientY }, zoom));

        event.currentTarget.setPointerCapture(event.pointerId);

        setStartPoint(mouse);
        setBoundingBox({ type: 'rectangle', x: mouse.x, y: mouse.y, width: 0, height: 0 });
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
        <SvgToolCanvas
            image={image}
            canvasRef={ref}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerDown={onPointerDown}
            style={{
                cursor: `url(${selectionCursor}) ${CURSOR_OFFSET}, auto`,
            }}
        >
            {boundingBox ? (
                <Rectangle
                    ariaLabel={'bounding box'}
                    rect={boundingBox}
                    styles={{ role: 'application', ...DEFAULT_ANNOTATION_STYLES }}
                />
            ) : null}
            <Crosshair location={crosshair.location} zoom={zoom} />
        </SvgToolCanvas>
    );
};
