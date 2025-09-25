// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isEmpty } from 'lodash-es';

import { Annotation as AnnotationType } from '../types';
import { DEFAULT_ANNOTATION_STYLES } from '../utils';
import { Annotation } from './annotation.component';
import { MaskAnnotations } from './mask-annotations.component';

type AnnotationsProps = {
    annotations: AnnotationType[];
    width: number;
    height: number;
    isFocussed: boolean;
};

export const Annotations = ({ annotations, width, height, isFocussed }: AnnotationsProps) => {
    return (
        <svg
            aria-label={'annotations'}
            width={width}
            height={height}
            style={{ position: 'absolute', inset: 0, ...DEFAULT_ANNOTATION_STYLES }}
        >
            {!isEmpty(annotations) && (
                <MaskAnnotations annotations={annotations} width={width} height={height} isEnabled={isFocussed}>
                    {annotations.map((annotation) => (
                        <Annotation annotation={annotation} key={annotation.id} />
                    ))}
                </MaskAnnotations>
            )}
        </svg>
    );
};
