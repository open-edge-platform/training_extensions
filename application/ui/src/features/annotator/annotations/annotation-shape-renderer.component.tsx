// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Annotation } from '../../../shared/types';
import { useCanvasSettings } from '../../dataset/media-preview/primary-toolbar/settings/canvas-settings-provider.component';
import { AnnotationShapeWithLabels } from './annotation-shape-with-labels.component';
import { AnnotationShapeWithoutLabels } from './annotation-shape-without-labels.component';

interface AnnotationShapeRendererProps {
    annotation: Annotation;
}

export const AnnotationShapeRenderer = ({ annotation }: AnnotationShapeRendererProps) => {
    const { canvasSettings } = useCanvasSettings();

    if (canvasSettings.hideLabels.value) {
        return <AnnotationShapeWithoutLabels annotation={annotation} />;
    }

    return <AnnotationShapeWithLabels annotation={annotation} />;
};
