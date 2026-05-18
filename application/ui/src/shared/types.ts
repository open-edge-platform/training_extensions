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
type FullImage = components['schemas']['FullImage'];

export type Shape = Rect | Polygon | FullImage;

export type AnnotationLabel = components['schemas']['LabelView'] & { probability?: number };

export type Annotation = Omit<components['schemas']['DatasetItemAnnotation-Input'], 'labels'> & {
    id: string;
    labels: AnnotationLabel[];
};

export type ClipperPoint = {
    X: number;
    Y: number;
};
