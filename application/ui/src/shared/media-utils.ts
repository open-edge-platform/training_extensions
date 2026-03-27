// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import * as UTIF from 'utif';

const TIFF_FORMATS = ['tif', 'tiff'];

export const isTiffFormat = (media: { format?: string }): boolean => {
    return TIFF_FORMATS.includes(String(media.format).toLowerCase());
};

export const getImageDataFromTiffUrl = async (url: string): Promise<ImageData> => {
    try {
        const response = await fetch(url);
        const buffer = await response.arrayBuffer();
        const ifds = UTIF.decode(buffer);

        if (ifds.length === 0) {
            return new ImageData(1, 1);
        }

        const tiff = ifds[0];
        UTIF.decodeImage(buffer, tiff);
        const rgba = UTIF.toRGBA8(tiff);

        return new ImageData(new Uint8ClampedArray(rgba), tiff.width, tiff.height);
    } catch (error) {
        console.error('Failed to decode TIFF image:', error);

        return new ImageData(1, 1);
    }
};
