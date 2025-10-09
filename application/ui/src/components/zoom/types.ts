// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export type ZoomState = {
    scale: number;
    maxZoomIn: number;
    translate: { x: number; y: number };
    initialCoordinates: { scale: number; x: number; y: number };
};
