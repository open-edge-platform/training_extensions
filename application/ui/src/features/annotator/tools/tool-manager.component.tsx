// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useAnnotator } from '../../../shared/annotator/annotator-provider.component';
import { BoundingBoxTool } from './bounding-box-tool/bounding-box-tool.component';
import { PolygonTool } from './polygon-tool/polygon-tool.component';
import { SegmentAnythingTool } from './segment-anything-tool/segment-anything-tool.component';

export const ToolManager = () => {
    const { activeTool } = useAnnotator();

    if (activeTool === 'bounding-box') {
        return <BoundingBoxTool />;
    }

    if (activeTool === 'sam') {
        return <SegmentAnythingTool />;
    }

    if (activeTool === 'polygon') {
        return <PolygonTool />;
    }

    return null;
};
