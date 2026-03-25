// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { components } from '../api/openapi-spec';

export interface RegionOfInterest {
    x: number;
    y: number;
    width: number;
    height: number;
}

export type Point = components['schemas']['Point'];
export type Rect = components['schemas']['Rectangle'];
export type Polygon = components['schemas']['Polygon'];
export type FullImage = components['schemas']['FullImage'];

export type Shape = Rect | Polygon | FullImage;

export type AnnotationLabel = components['schemas']['LabelView'] & { probability?: number };

export type Annotation = Omit<components['schemas']['DatasetItemAnnotation-Input'], 'labels'> & {
    id: string;
    labels: AnnotationLabel[];
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
