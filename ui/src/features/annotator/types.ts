// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { components } from '../../api/openapi-spec';

export interface RegionOfInterest {
    x: number;
    y: number;
    width: number;
    height: number;
}

export interface Point {
    x: number;
    y: number;
}

export interface Rect {
    readonly shapeType: 'rect';
    readonly x: number;
    readonly y: number;
    readonly width: number;
    readonly height: number;
}

export interface Polygon {
    readonly shapeType: 'polygon';
    readonly points: Point[];
}

export type Shape = Rect | Polygon;

export type Label = { id: string; name: string; color: string; isPrediction: boolean; score?: number };

export type Annotation = {
    id: string;
    shape: Shape;
    labels: Array<Label>;
};

export type AnnotationState = {
    isHovered: boolean;
    isSelected: boolean;
    isHidden: boolean;
    isLocked: boolean;
};

// Circle is only used for visual purposes on segment-anything tool
export interface Circle {
    readonly shapeType: 'circle';
    readonly x: number;
    readonly y: number;
    readonly r: number;
}

export interface ClipperPoint {
    X: number;
    Y: number;
}
export type DatasetItem = components['schemas']['DatasetItem'];
