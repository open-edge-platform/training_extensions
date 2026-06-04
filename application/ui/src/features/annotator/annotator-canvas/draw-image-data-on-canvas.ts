// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export const drawImageDataOnCanvas = (ctx: CanvasRenderingContext2D, image: ImageData): boolean => {
    if (image.data.length !== image.width * image.height * 4) {
        return false;
    }

    const compatibleImageData = ctx.createImageData(image.width, image.height);
    compatibleImageData.data.set(image.data);
    ctx.putImageData(compatibleImageData, 0, 0);

    return true;
};
