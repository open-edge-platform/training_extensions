// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { MediaVideoFrame } from '../../../constants/shared-types';
import { Annotation } from '../../../shared/types';
import { DEFAULT_ANNOTATION_STYLES } from '../utils';
import { AnnotationShapeRenderer } from './annotation-shape-renderer.component';

type VideoAnnotationsProps = {
    videoFrame: MediaVideoFrame;
};

export const VideoAnnotations = ({ videoFrame }: VideoAnnotationsProps) => {
    const annotations: Annotation[] = [];

    return (
        <svg
            aria-label={'video annotations'}
            data-testid={'annotation-layer'}
            width={videoFrame.width}
            height={videoFrame.height}
            tabIndex={-1}
            style={{
                position: 'absolute',
                inset: 0,
                outline: 'none',
                overflow: 'visible',
                ...DEFAULT_ANNOTATION_STYLES,
            }}
        >
            {annotations.map((annotation) => (
                <AnnotationShapeRenderer key={annotation.id} annotation={annotation} />
            ))}
        </svg>
    );
};
