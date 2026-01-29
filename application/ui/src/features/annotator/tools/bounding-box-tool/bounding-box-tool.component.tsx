// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useZoom } from '../../../../components/zoom/zoom.provider';
import type { Label } from '../../../../constants/shared-types';
import { useAnnotationActions } from '../../../../shared/annotator/annotation-actions-provider.component';
import { useAnnotator } from '../../../../shared/annotator/annotator-provider.component';
import { useSelectedAnnotations } from '../../../../shared/annotator/select-annotation-provider.component';
import type { Rect } from '../../../../shared/types';
import { DrawingBox } from '../drawing-box-tool/drawing-box.component';

export const BoundingBoxTool = () => {
    const { scale: zoom } = useZoom();
    const { addAnnotations } = useAnnotationActions();
    const { setSelectedAnnotations } = useSelectedAnnotations();
    const { mediaItem, image, selectedLabel } = useAnnotator();

    const handleComplete = (shapes: Rect[], labels: Label[]): string[] => {
        const newIds = addAnnotations(shapes, labels);

        setSelectedAnnotations(new Set(newIds));

        return newIds;
    };

    return (
        <DrawingBox
            roi={{ x: 0, y: 0, width: mediaItem.width, height: mediaItem.height }}
            image={image}
            zoom={zoom}
            selectedLabel={selectedLabel}
            onComplete={handleComplete}
        />
    );
};
