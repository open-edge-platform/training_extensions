// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useZoom } from '../../../../components/zoom/zoom';
import { Annotation } from '../../types';
import { TranslateShape } from '../bounding-box-tool/translate-shape.component';

type SelectionToolProps = {
    annotation: Annotation;
    updateAnnotation: (annotation: Annotation) => void;
};

export const SelectionTool = ({ annotation, updateAnnotation }: SelectionToolProps) => {
    const { scale } = useZoom();

    const translate = ({ x, y }: { x: number; y: number }) => {
        const shape = annotation.shape;

        if (shape.shapeType === 'rect') {
            const newShape = { ...shape, x: shape.x + x, y: shape.y + y };

            updateAnnotation({ ...annotation, shape: newShape });

            return;
        }
    };

    return (
        <TranslateShape
            zoom={scale}
            annotation={annotation}
            translateShape={translate}
            onComplete={() => console.log('complete')}
        />
    );
};
