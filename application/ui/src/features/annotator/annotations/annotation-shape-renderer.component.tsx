// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCanvasSettings } from '../../dataset/media-preview/primary-toolbar/settings/canvas-settings-provider.component';
import { Annotation } from '../types';
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
