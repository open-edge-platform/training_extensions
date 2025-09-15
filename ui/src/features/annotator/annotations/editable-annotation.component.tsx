// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { useZoom } from '../../../components/zoom/zoom';
import { useSelectedAnnotations } from '../select-annotation-provider.component';
import { EditBoundingBox } from '../tools/bounding-box-tool/bounding-box-tool.component';
import { Annotation, Rect } from '../types';
import { useAnnotation } from './annotation.component';

interface EditAnnotationProps {
    children: ReactNode;
}

export const EditableAnnotation = ({ children }: EditAnnotationProps) => {
    const annotation = useAnnotation() as Annotation & { shape: Rect };
    const { selectedAnnotations } = useSelectedAnnotations();
    const { scale } = useZoom();

    const { shape } = annotation;

    const isSelected = selectedAnnotations.has(annotation.id);

    if (isSelected && selectedAnnotations.size === 1) {
        return (
            <EditBoundingBox
                key={`bbox-${shape.x}-${shape.y}-${shape.width}-${shape.height}`}
                annotation={annotation}
                zoom={scale}
            />
        );
    }

    return <>{children}</>;
};
