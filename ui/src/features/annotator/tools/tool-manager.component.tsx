// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useZoom } from '../../../components/zoom/zoom';
import { useAnnotator } from '../annotator-provider.component';
import { Annotation, Rect } from '../types';
import { EditBoundingBox } from './bounding-box-tool/bounding-box-tool.component';
import { SelectionTool } from './selection-tool/selection-tool.component';

type ToolManager = {
    width: number;
    height: number;
};
export const ToolManager = ({ width, height }: ToolManager) => {
    const zoom = useZoom();
    const { selectedAnnotation, updateAnnotation, activeTool } = useAnnotator();

    if (!selectedAnnotation) {
        return null;
    }

    if (activeTool === 'bounding-box') {
        return (
            <EditBoundingBox
                annotation={selectedAnnotation as Annotation & { shape: Rect }}
                roi={{ x: 0, y: 0, width, height }}
                zoom={zoom.scale}
                updateAnnotation={updateAnnotation}
            />
        );
    }

    if (activeTool === 'selection') {
        return <SelectionTool annotation={selectedAnnotation} updateAnnotation={updateAnnotation} />;
    }

    return null;
};
