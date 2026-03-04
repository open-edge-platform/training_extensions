// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { mapServerAnnotationsToLocal } from '../../../shared/annotator/annotation-mappers';
import { useProjectLabelsWithEmptyLabel } from '../../../shared/annotator/labels';
import { DEFAULT_ANNOTATION_STYLES } from '../utils';
import { useVideoFramesAnnotations } from '../video-player/api/use-video-frames-annotations';
import { useVideoPlayer } from '../video-player/video-player-provider.component';
import { AnnotationShapeRenderer } from './annotation-shape-renderer.component';

export const VideoAnnotations = () => {
    const { step, videoFrame } = useVideoPlayer();

    const labels = useProjectLabelsWithEmptyLabel();

    const { data: annotations = [] } = useVideoFramesAnnotations({
        frameNumber: videoFrame.frame_number,
        frameSkip: step,
        selector: (data) => {
            const frameAnnotations =
                data.find((frame) => frame.frame_index === videoFrame.frame_number)?.annotation_data?.annotations ?? [];

            return mapServerAnnotationsToLocal(frameAnnotations, labels);
        },
    });

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
