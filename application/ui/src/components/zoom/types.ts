// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export type Point = {
    x: number;
    y: number;
};

export type ZoomState = {
    scale: number;
    maxZoomIn: number;
    translate: Point;
    hasAnimation: boolean;
    initialCoordinates: Point & { scale: number };
};
