// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { AnnotationShape } from '../../annotations/annotation-shape.component';
import { Annotation } from '../../types';

type SelectionToolProps = {
    annotation: Annotation & { shape: { shapeType: 'rect' } };
};

export const SelectionTool = ({ annotation }: SelectionToolProps) => {
    return (
        <svg
            width={annotation.shape.width}
            height={annotation.shape.height}
            style={{ inset: 0, position: 'absolute' }}
            id={`select-shape-${annotation.id}`}
        >
            <AnnotationShape annotation={annotation} />
        </svg>
    );
};
