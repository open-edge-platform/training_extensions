// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useZoom } from '../../../../components/zoom/zoom.provider';
import { useAnnotationActions } from '../../annotation-actions-provider.component';
import { useAnnotator } from '../../annotator-provider.component';
import { DrawingBox } from '../drawing-box-tool/drawing-box.component';

export const BoundingBoxTool = () => {
    const { mediaItem, image } = useAnnotator();
    const { addAnnotations } = useAnnotationActions();
    const { scale: zoom } = useZoom();

    return (
        <DrawingBox
            roi={{ x: 0, y: 0, width: mediaItem.width, height: mediaItem.height }}
            image={image}
            zoom={zoom}
            onComplete={addAnnotations}
        />
    );
};
