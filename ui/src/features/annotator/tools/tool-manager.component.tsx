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

    // We only want to allow edition if there is exactly one selected annotation selected
    if (currentlySelectedAnnotations.length === 0 || currentlySelectedAnnotations.length > 1) {
        return null;
    }

    const annotation = currentlySelectedAnnotations[0] as Annotation & { shape: Rect };
    const { shape } = annotation;

    if (activeTool === 'selection') {
        return (
            <EditBoundingBox
                key={`bbox-${shape.x}-${shape.y}-${shape.width}-${shape.height}`}
                annotation={annotation}
                roi={{ x: 0, y: 0, width, height }}
                zoom={zoom.scale}
                updateAnnotation={updateAnnotation}
            />
        );
    }

    return null;
};
