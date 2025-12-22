// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useZoom } from '../../../../components/zoom/zoom.provider';
import { useAnnotationActions } from '../../../../shared/annotator/annotation-actions-provider.component';
import { useAnnotator } from '../../../../shared/annotator/annotator-provider.component';
import { DrawingBox } from '../drawing-box-tool/drawing-box.component';

export const BoundingBoxTool = () => {
    const { mediaItem, image, selectedLabel } = useAnnotator();
    const { addAnnotations } = useAnnotationActions();
    const { scale: zoom } = useZoom();

    return (
        <DrawingBox
            roi={{ x: 0, y: 0, width: mediaItem.width, height: mediaItem.height }}
            image={image}
            zoom={zoom}
            selectedLabel={selectedLabel}
            onComplete={addAnnotations}
        />
    );
};
