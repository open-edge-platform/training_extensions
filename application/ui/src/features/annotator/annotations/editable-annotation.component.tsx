// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { useZoom } from '../../../components/zoom/zoom.provider';
import { useSelectedAnnotations } from '../../../shared/annotator/select-annotation-provider.component';
import { EditBoundingBox } from '../tools/edit-bounding-box/edit-bounding-box.component';
import { EditPolygon } from '../tools/edit-polygon/edit-polygon.component';
import { useAnnotation } from './annotation-context';
import { isPolygon, isRectangle } from './utils';

interface EditAnnotationProps {
    children: ReactNode;
}

export const EditableAnnotation = ({ children }: EditAnnotationProps) => {
    const { scale } = useZoom();
    const annotation = useAnnotation();
    const { selectedAnnotations } = useSelectedAnnotations();

    const isSelected = selectedAnnotations.has(annotation.id);

    if (isSelected && selectedAnnotations.size === 1) {
        if (isPolygon(annotation)) {
            return <EditPolygon annotation={annotation} zoom={scale} />;
        }

        if (isRectangle(annotation)) {
            const { shape } = annotation;
            return (
                <EditBoundingBox
                    key={`box-${shape.x}-${shape.y}-${shape.width}-${shape.height}`}
                    annotation={annotation}
                    zoom={scale}
                />
            );
        }
    }

    return <>{children}</>;
};
