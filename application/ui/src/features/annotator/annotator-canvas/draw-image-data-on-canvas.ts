// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isImageOversized } from '../tools/utils';

// Draw ImageData to canvas. Fails gracefully for oversized media (downscaled data).
export const drawImageDataOnCanvas = (ctx: CanvasRenderingContext2D, image: ImageData): boolean => {
    if (isImageOversized(image)) return false;

    const compatibleImageData = ctx.createImageData(image.width, image.height);
    compatibleImageData.data.set(image.data);
    ctx.putImageData(compatibleImageData, 0, 0);

    return true;
};
