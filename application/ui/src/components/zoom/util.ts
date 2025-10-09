// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ZoomState } from './types';

export const ZOOM_STEP_DIVISOR = 10;

export const getZoomState =
    ({
        initialCoordinates,
        newScale,
        cursorX,
        cursorY,
    }: {
        newScale: number;
        cursorX: number;
        cursorY: number;
        initialCoordinates: ZoomState['initialCoordinates'];
    }) =>
    (prev: ZoomState) => {
        if (newScale <= initialCoordinates.scale) {
            return {
                ...prev,
                scale: initialCoordinates.scale,
                translate: { x: initialCoordinates.x, y: initialCoordinates.y },
            };
        }

        const scaleRatio = newScale / prev.scale;
        const newTranslateX = cursorX - scaleRatio * (cursorX - prev.translate.x);
        const newTranslateY = cursorY - scaleRatio * (cursorY - prev.translate.y);

        return {
            ...prev,
            scale: newScale,
            translate: { x: newTranslateX, y: newTranslateY },
        };
    };
