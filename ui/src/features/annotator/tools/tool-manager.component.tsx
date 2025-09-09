// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useZoom } from '../../../components/zoom/zoom';
import { useAnnotator } from '../annotator-provider.component';
import { useSelectedAnnotations } from '../select-annotation-provider.component';
import { Annotation, Rect } from '../types';
import { EditBoundingBox } from './bounding-box-tool/bounding-box-tool.component';

type ToolManager = {
    width: number;
    height: number;
};
export const ToolManager = ({ width, height }: ToolManager) => {
    const zoom = useZoom();
    const { updateAnnotation, activeTool, annotations } = useAnnotator();
    const selectedAnnotations = useSelectedAnnotations();

    const currentlySelectedAnnotations = annotations.filter((annotation) => selectedAnnotations.has(annotation.id));

    if (currentlySelectedAnnotations.length === 0) {
        return null;
    }

    if (activeTool === 'selection') {
        return (
            <EditBoundingBox
                annotation={currentlySelectedAnnotations[0] as Annotation & { shape: Rect }}
                roi={{ x: 0, y: 0, width, height }}
                zoom={zoom.scale}
                updateAnnotation={updateAnnotation}
            />
        );
    }

    return null;
};
