// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { drawImageDataOnCanvas } from './draw-image-data-on-canvas';

const createCanvasContext = () => {
    return {
        createImageData: vi.fn((width: number, height: number) => ({
            width,
            height,
            data: new Uint8ClampedArray(width * height * 4),
            colorSpace: 'srgb',
        })),
        putImageData: vi.fn(),
    } as unknown as CanvasRenderingContext2D;
};

describe('drawImageDataOnCanvas', () => {
    it('skips drawing when the input pixel buffer is invalid', () => {
        const ctx = createCanvasContext();
        const invalidImage = {
            width: 4,
            height: 4,
            data: new Uint8ClampedArray(10),
            colorSpace: 'srgb',
        } as ImageData;

        const didDraw = drawImageDataOnCanvas(ctx, invalidImage);

        expect(didDraw).toBe(false);
        expect(ctx.createImageData).not.toHaveBeenCalled();
        expect(ctx.putImageData).not.toHaveBeenCalled();
    });

    it('normalizes and draws valid image-shaped data', () => {
        const ctx = createCanvasContext();
        const image = {
            width: 2,
            height: 3,
            data: new Uint8ClampedArray(2 * 3 * 4).fill(7),
            colorSpace: 'srgb',
        } as ImageData;

        const didDraw = drawImageDataOnCanvas(ctx, image);

        const created = vi.mocked(ctx.createImageData).mock.results[0]?.value as ImageData;

        expect(didDraw).toBe(true);
        expect(ctx.createImageData).toHaveBeenCalledWith(2, 3);
        expect(created.data).toEqual(image.data);
        expect(ctx.putImageData).toHaveBeenCalledWith(created, 0, 0);
    });
});
