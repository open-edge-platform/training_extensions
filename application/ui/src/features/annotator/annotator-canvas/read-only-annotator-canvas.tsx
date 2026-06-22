// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Media } from '../../../constants/shared-types';
import { ReadOnlyAnnotations } from '../annotations/read-only-annotations.component';
import { MediaCanvas } from './media-canvas';

type ReadOnlyAnnotatorCanvasProps = {
    mediaItem: Media;
    image: ImageData;
};

export const ReadOnlyAnnotatorCanvas = ({ mediaItem, image }: ReadOnlyAnnotatorCanvasProps) => (
    <MediaCanvas mediaItem={mediaItem} image={image}>
        <ReadOnlyAnnotations width={mediaItem.width} height={mediaItem.height} />
    </MediaCanvas>
);
