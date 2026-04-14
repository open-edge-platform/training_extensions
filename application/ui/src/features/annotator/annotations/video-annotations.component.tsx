// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { mapServerAnnotationsToLocal } from '../../../shared/annotator/annotation-mappers';
import { useProjectLabelsWithEmptyLabel } from '../../../shared/annotator/labels';
import { Annotation } from '../../../shared/types';
import { DEFAULT_ANNOTATION_STYLES } from '../utils';
import { useVideoFramesAnnotations } from '../video-player/api/use-video-frames-annotations';
import {
    PREDICTION_CHUNK_SIZE,
    PREDICTION_FRAME_SKIP,
    useVideoFramesPredictions,
} from '../video-player/api/use-video-frames-predictions';
import { useVideoPlayer } from '../video-player/video-player-provider.component';
import { AnnotationShapeRenderer } from './annotation-shape-renderer.component';

type AnnotationsRendererProps = {
    annotations: Annotation[];
    height: number;
    width: number;
    ariaLabel: string;
};

const AnnotationsRenderer = ({ annotations, width, height, ariaLabel }: AnnotationsRendererProps) => {
    return (
        <svg
            aria-label={ariaLabel}
            data-testid={'annotation-layer'}
            width={width}
            height={height}
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
        <AnnotationsRenderer
            annotations={annotations}
            height={videoFrame.height}
            width={videoFrame.width}
            ariaLabel={'video annotations'}
        />
    );
};

export const VideoPredictions = () => {
    const { videoFrame } = useVideoPlayer();

    const labels = useProjectLabelsWithEmptyLabel();
    const { data: predictions = [] } = useVideoFramesPredictions({
        frameNumber: videoFrame.frame_number,
        frameSkip: PREDICTION_FRAME_SKIP,
        chunkSize: PREDICTION_CHUNK_SIZE,
        selector: (data) => {
            const idxToPredictionsMap = new Map(data.map((frame) => [frame.media.frame_index, frame.prediction]));

            if (idxToPredictionsMap.has(videoFrame.frame_number)) {
                return mapServerAnnotationsToLocal(idxToPredictionsMap.get(videoFrame.frame_number) ?? [], labels);
            }

            for (let i = 0; i < PREDICTION_CHUNK_SIZE; i++) {
                if (idxToPredictionsMap.has(videoFrame.frame_number + i)) {
                    return mapServerAnnotationsToLocal(
                        idxToPredictionsMap.get(videoFrame.frame_number + i) ?? [],
                        labels
                    );
                } else if (idxToPredictionsMap.has(videoFrame.frame_number - i)) {
                    return mapServerAnnotationsToLocal(
                        idxToPredictionsMap.get(videoFrame.frame_number - i) ?? [],
                        labels
                    );
                }
            }

            return mapServerAnnotationsToLocal([], labels);
        },
    });

    return (
        <AnnotationsRenderer
            annotations={predictions}
            height={videoFrame.height}
            width={videoFrame.width}
            ariaLabel={'video predictions'}
        />
    );
};
