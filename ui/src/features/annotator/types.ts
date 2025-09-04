// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export type BoundingBox = {
    type: 'bounding-box';
    x: number;
    y: number;
    width: number;
    height: number;
};

export type Point = { x: number; y: number };
export type Polygon = {
    type: 'polygon';
    points: Array<Point>;
};

export type Shape = BoundingBox | Polygon;

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
