// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, SVGProps } from 'react';

import { isEmpty, isEqual, negate } from 'lodash-es';

import { Label } from '../../../../constants/shared-types';
import { isEraserOrRightButton, isLeftButton } from '../../buttons-utils';
import { Point } from '../../types';
import { DEFAULT_ANNOTATION_STYLES } from '../../utils';
import { PointerType } from '../utils';

export interface ShapeStyle<T> {
    styles?: SVGProps<T>;
    className?: string;
}

export enum PolygonMode {
    Eraser = 'eraser',
    Lasso = 'lasso',
    LassoClose = 'lassoClose',
    Polygon = 'polygon',
    PolygonClose = 'polygonClose',
    MagneticLasso = 'magneticLasso',
    MagneticLassoClose = 'magneticLassoClose',
}

export enum PointerIcons {
    Eraser = 'eraser-tool',
    Lasso = 'lasso-drawing',
    LassoClose = 'lasso-closing',
    Polygon = 'polygon-drawing',
    PolygonClose = 'polygon-closing',
    MagneticLasso = 'magnetic-lasso-drawing',
    MagneticLassoClose = 'magnetic-lasso-closing',
}

export enum PointerIconsOffset {
    Eraser = '15 15',
    Lasso = '0 0',
    LassoClose = '0 0',
    Polygon = '0 0',
    PolygonClose = '0 0',
    MagneticLasso = '0 0',
    MagneticLassoClose = '0 0',
}

export const TOOL_ICON: Record<PolygonMode, { icon: PointerIcons; offset: PointerIconsOffset }> = {
    [PolygonMode.Lasso]: { icon: PointerIcons.Lasso, offset: PointerIconsOffset.Lasso },
    [PolygonMode.Eraser]: { icon: PointerIcons.Eraser, offset: PointerIconsOffset.Eraser },
    [PolygonMode.Polygon]: { icon: PointerIcons.Polygon, offset: PointerIconsOffset.Polygon },
    [PolygonMode.LassoClose]: { icon: PointerIcons.LassoClose, offset: PointerIconsOffset.LassoClose },
    [PolygonMode.MagneticLasso]: { icon: PointerIcons.MagneticLasso, offset: PointerIconsOffset.MagneticLasso },
    [PolygonMode.MagneticLassoClose]: {
        icon: PointerIcons.MagneticLassoClose,
        offset: PointerIconsOffset.MagneticLassoClose,
    },
    [PolygonMode.PolygonClose]: { icon: PointerIcons.PolygonClose, offset: PointerIconsOffset.PolygonClose },
};

export const ERASER_FIELD_DEFAULT_RADIUS = 5;
export const START_POINT_FIELD_DEFAULT_RADIUS = 6;
export const START_POINT_FIELD_FOCUS_RADIUS = 8;

const isDifferent = negate(isEqual);

export const getCloseMode = (mode: PolygonMode | null) => {
    if (mode === PolygonMode.MagneticLasso) {
        return PolygonMode.MagneticLassoClose;
    }

    if (mode === PolygonMode.Lasso) {
        return PolygonMode.LassoClose;
    }

    return PolygonMode.PolygonClose;
};

export const drawingStyles = (defaultLabel: Label | null): typeof DEFAULT_ANNOTATION_STYLES => {
    if (defaultLabel === null) {
        return DEFAULT_ANNOTATION_STYLES;
    }

    return {
        ...DEFAULT_ANNOTATION_STYLES,
        fill: defaultLabel.color,
        stroke: defaultLabel.color,
    };
};

export const getToolIcon = (polygonMode: PolygonMode | null) => {
    if (polygonMode === null) {
        return TOOL_ICON[PolygonMode.Polygon];
    }

    return TOOL_ICON[polygonMode];
};

export const getFormattedPoints = (points: Point[]): string => points.map(({ x, y }) => `${x},${y}`).join(' ');

type PointerSVGElement = PointerEvent<SVGElement>;
type CallbackPointerVoid = (event: PointerSVGElement) => void;

export interface MouseButton {
    button: number;
    buttons: number;
}

const mouseButtonEventValidation =
    (callback: CallbackPointerVoid) => (predicate: (button: MouseButton) => boolean) => (event: PointerSVGElement) => {
        event.preventDefault();

        if (event.pointerType === PointerType.Touch) return;

        const button = {
            button: event.button,
            buttons: event.buttons,
        };

        if (predicate(button)) {
            callback(event);
        }
    };

export const leftMouseButtonHandler = (callback: CallbackPointerVoid): CallbackPointerVoid =>
    mouseButtonEventValidation(callback)(isLeftButton);

export const rightMouseButtonHandler = (callback: CallbackPointerVoid): CallbackPointerVoid =>
    mouseButtonEventValidation(callback)(isEraserOrRightButton);

export const leftRightMouseButtonHandler =
    (leftCallback: CallbackPointerVoid, rightCallback: CallbackPointerVoid) =>
    (event: PointerSVGElement): void => {
        leftMouseButtonHandler(leftCallback)(event);
        rightMouseButtonHandler(rightCallback)(event);
    };

export interface MouseEventHandlers {
    onPointerUp: (event: PointerEvent<SVGSVGElement>) => void;
    onPointerDown: (event: PointerEvent<SVGSVGElement>) => void;
    onPointerMove: (event: PointerEvent<SVGSVGElement>) => void;
}

export const removeEmptySegments =
    (...newSegments: Point[][]) =>
    (prevSegments: Point[][]): Point[][] => {
        const validSegments = newSegments.filter(negate(isEmpty));

        return isEmpty(validSegments) ? [...prevSegments] : [...prevSegments, ...validSegments];
    };

export const deleteSegments =
    (intersectionPoint: Point) =>
    (segments: Point[][]): Point[][] => {
        return segments
            .map((segment: Point[]) => segment.filter((point: Point) => isDifferent(point, intersectionPoint)))
            .filter(negate(isEmpty));
    };

export const isCloseMode = (mode: PolygonMode | null) => {
    if (mode == null) {
        return false;
    }

    return [PolygonMode.PolygonClose, PolygonMode.LassoClose, PolygonMode.MagneticLassoClose].includes(mode);
};
