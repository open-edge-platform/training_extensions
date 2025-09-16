// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, RefObject, useState } from 'react';

import { Point } from '../../../types';

interface UseCrosshair {
    location: Point | null;
    onPointerMove: (event: PointerEvent<SVGSVGElement>) => void;
    onPointerLeave: (event: PointerEvent<SVGSVGElement>) => void;
}

type ElementType = SVGElement | HTMLDivElement;
export const getRelativePoint = (element: ElementType, point: Point, zoom: number): Point => {
    const rect = element.getBoundingClientRect();

    return {
        x: Math.round((point.x - rect.left) / zoom),
        y: Math.round((point.y - rect.top) / zoom),
    };
};

export const useCrosshair = (canvasRef: RefObject<SVGRectElement | null>, zoom: number): UseCrosshair => {
    const [location, setLocation] = useState<Point | null>(null);

    const onPointerMove = (event: PointerEvent<SVGSVGElement>) => {
        if (canvasRef.current === null) {
            return;
        }

        const newLocation = getRelativePoint(canvasRef.current, { x: event.clientX, y: event.clientY }, zoom);

        setLocation(newLocation);
    };

    const onPointerLeave = (_event: PointerEvent<SVGSVGElement>) => {
        setLocation(null);
    };

    return { location, onPointerMove, onPointerLeave };
};
