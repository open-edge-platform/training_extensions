// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { components } from '../../api/openapi-spec';

export interface RegionOfInterest {
    x: number;
    y: number;
    width: number;
    height: number;
}

export type Point = components['schemas']['Point'];
export type Rect = components['schemas']['Rectangle'];
export type Polygon = components['schemas']['Polygon'];

export type Shape = Rect | Polygon;

export type Label = components['schemas']['Label'] & {
    id: string;
    name: string;
    color: string;
};

export type Annotation = components['schemas']['DatasetItemAnnotation-Input'] & {
    id: string;
    labels: Label[];
};

export type AnnotationState = {
    isHovered: boolean;
    isSelected: boolean;
    isHidden: boolean;
    isLocked: boolean;
};

// Circle is only used for visual purposes on segment-anything tool
export type Circle = {
    readonly type: 'circle';
    readonly x: number;
    readonly y: number;
    readonly r: number;
};

export type ClipperPoint = {
    X: number;
    Y: number;
};
export type DatasetItem = components['schemas']['DatasetItem'];
