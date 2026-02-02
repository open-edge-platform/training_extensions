// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useZoom } from '../../../../components/zoom/zoom.provider';
import type { Label } from '../../../../constants/shared-types';
import { useAnnotator } from '../../../../shared/annotator/annotator-provider.component';
import type { Rect } from '../../../../shared/types';
import { DrawingBox } from '../drawing-box-tool/drawing-box.component';
import { useAddAndSelectAnnotations } from '../use-add-and-select-annotations.hook';

export const BoundingBoxTool = () => {
    const { scale: zoom } = useZoom();
    const { addAndSelectAnnotations } = useAddAndSelectAnnotations();
    const { mediaItem, image, selectedLabel } = useAnnotator();

    const handleComplete = (shapes: Rect[], labels: Label[]): string[] => {
        return addAndSelectAnnotations(shapes, labels);
    };

    return (
        <DrawingBox
            roi={{ x: 0, y: 0, width: mediaItem.width ?? 0, height: mediaItem.height ?? 0 }}
            image={image}
            zoom={zoom}
            selectedLabel={selectedLabel}
            onComplete={handleComplete}
        />
    );
};
