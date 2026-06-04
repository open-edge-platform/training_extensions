// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { canRasteriseAtFullSize, getImageData } from './utils';

const createImage = (width: number, height: number, naturalWidth = width, naturalHeight = height): HTMLImageElement =>
    ({ width, height, naturalWidth, naturalHeight }) as HTMLImageElement;

const createCanvasContext = () => {
    return {
        filter: '',
        drawImage: vi.fn(),
        getImageData: vi.fn((_x: number, _y: number, width: number, height: number) => {
            return {
                width,
                height,
                data: new Uint8ClampedArray(width * height * 4),
                colorSpace: 'srgb',
            } as ImageData;
        }),
    } as unknown as CanvasRenderingContext2D;
};

const mockCanvasCreation = (contexts: Array<CanvasRenderingContext2D | null>) => {
    const originalCreateElement = document.createElement.bind(document);

    return vi.spyOn(document, 'createElement').mockImplementation(((tagName: string) => {
        if (tagName !== 'canvas') {
            return originalCreateElement(tagName as keyof HTMLElementTagNameMap);
        }

        const context = contexts.shift() ?? null;
        return {
            width: 0,
            height: 0,
            getContext: vi.fn(() => context),
        } as unknown as HTMLCanvasElement;
    }) as typeof document.createElement);
};

afterEach(() => {
    vi.restoreAllMocks();
});

describe('canRasteriseAtFullSize', () => {
    it('accepts the exact boundary size 16384x16384', () => {
        expect(canRasteriseAtFullSize(16384, 16384)).toBe(true);
    });

    it('rejects dimensions one pixel over the boundary', () => {
        expect(canRasteriseAtFullSize(16385, 16384)).toBe(false);
        expect(canRasteriseAtFullSize(16384, 16385)).toBe(false);
    });
});

describe('getImageData', () => {
    it('returns 1x1 placeholder for empty images', () => {
        const imageData = getImageData(createImage(0, 0));

        expect(imageData.width).toBe(1);
        expect(imageData.height).toBe(1);
    });

    it('downscales oversized images to max side 4096', () => {
        const context = createCanvasContext();
        mockCanvasCreation([context]);

        const imageData = getImageData(createImage(19156, 15010));
        const drawImageArgs = vi.mocked(context.drawImage).mock.calls[0];

        const decodedWidth = drawImageArgs[3] as number;
        const decodedHeight = drawImageArgs[4] as number;

        expect(Math.max(decodedWidth, decodedHeight)).toBe(4096);
        expect([imageData.width, imageData.height].sort((a, b) => a - b)).toEqual([3209, 4096]);
    });

    it('falls back to downscaled decode when full-size canvas context is unavailable', () => {
        const fallbackContext = createCanvasContext();
        const createElementSpy = mockCanvasCreation([null, fallbackContext]);

        const imageData = getImageData(createImage(4000, 3000));
        const fallbackArgs = vi.mocked(fallbackContext.drawImage).mock.calls[0];

        const decodedWidth = fallbackArgs[3] as number;
        const decodedHeight = fallbackArgs[4] as number;

        expect(createElementSpy).toHaveBeenCalledTimes(2);
        expect([decodedWidth, decodedHeight].sort((a, b) => a - b)).toEqual([3000, 4000]);
        expect([imageData.width, imageData.height].sort((a, b) => a - b)).toEqual([3000, 4000]);
    });
});
