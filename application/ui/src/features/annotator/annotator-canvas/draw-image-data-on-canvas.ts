// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isImageOversized } from '../tools/utils';

// Draw ImageData to canvas. Fails gracefully for oversized media (downscaled data).
export const drawImageDataOnCanvas = (ctx: CanvasRenderingContext2D, image: ImageData): boolean => {
    if (isImageOversized(image)) return false;

    ctx.putImageData(image, 0, 0);

    return true;
};
