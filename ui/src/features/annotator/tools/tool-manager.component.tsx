// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useZoom } from '../../../components/zoom/zoom';
import { useAnnotator } from '../annotator-provider.component';
import { DrawingBox } from './drawing-box-tool/drawing-box.component';

export const ToolManager = () => {
    const { activeTool, mediaItem } = useAnnotator();
    const { scale } = useZoom();

    const handleComplete = () => {
        console.log('complete');
    };

    if (activeTool === 'bounding-box') {
        return (
            <DrawingBox
                zoom={scale}
                onComplete={handleComplete}
                image={new ImageData(mediaItem.width, mediaItem.height)}
            />
        );
    }

    return null;
};
