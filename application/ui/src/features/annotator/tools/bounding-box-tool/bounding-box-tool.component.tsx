// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useZoom } from '../../../../components/zoom/zoom.provider';
import type { Label } from '../../../../constants/shared-types';
import type { Rect } from '../../../../shared/types';
import { useLabelsProvider } from '../../labels-provider.component';
import { useMediaItemImage, useSelectedMediaItem } from '../../selected-media-item-provider.component';
import { DrawingBox } from '../drawing-box-tool/drawing-box.component';
import { useAddAndSelectAnnotations } from '../use-add-and-select-annotations.hook';

export const BoundingBoxTool = () => {
    const { scale: zoom } = useZoom();
    const { addAndSelectAnnotations } = useAddAndSelectAnnotations();
    const { roi } = useSelectedMediaItem();
    const { image } = useMediaItemImage();
    const { selectedLabel } = useLabelsProvider();

    const handleComplete = (shapes: Rect[], labels: Label[]): string[] => {
        return addAndSelectAnnotations(shapes, labels);
    };

    return <DrawingBox roi={roi} image={image} zoom={zoom} selectedLabel={selectedLabel} onComplete={handleComplete} />;
};
