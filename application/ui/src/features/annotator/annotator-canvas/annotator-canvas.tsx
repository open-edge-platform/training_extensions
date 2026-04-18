// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { MouseEvent, useEffect, useRef } from 'react';

import { ZoomTransform } from '../../../components/zoom/zoom-transform';
import type { Media } from '../../../constants/shared-types';
import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';
import { useAnnotationVisibility } from '../../../shared/annotator/annotation-visibility-provider.component';
import type { AnnotatorMode } from '../../../shared/annotator/annotator-mode';
import { useSelectedAnnotations } from '../../../shared/annotator/select-annotation-provider.component';
import { isVideo, isVideoFrame } from '../../../shared/media-item-utils';
import { Annotations } from '../annotations/annotations.component';
import { VideoAnnotations, VideoPredictions } from '../annotations/video-annotations.component';
import { useIsAnnotatorSceneBusy } from '../hooks/use-is-annotator-scene-busy';
import { ToolManager } from '../tools/tool-manager.component';
import {
    PREDICTION_CHUNK_SIZE,
    PREDICTION_FRAME_SKIP,
    usePrefetchVideoFramesPredictions,
} from '../video-player/api/use-video-frames-predictions';
import { getVideoFrameRangeIndexes } from '../video-player/api/utils';
import { VideoFrame } from '../video-player/video-frame.component';
import { useVideoPlayer, useVideoPlayerContext } from '../video-player/video-player-provider.component';

import classes from './annotator-canvas.module.scss';

type MediaImageProps = {
    image: ImageData;
    mediaItem: Media;
};

const useDrawImageOnCanvas = (image: ImageData) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const ctx = canvasRef?.current?.getContext('2d');

        if (ctx == null) {
            return;
        }

        ctx.putImageData(image, 0, 0);
    }, [image]);

    return canvasRef;
};

const MediaImage = ({ image, mediaItem }: MediaImageProps) => {
    const canvasRef = useDrawImageOnCanvas(image);

    return (
        <>
            <canvas ref={canvasRef} width={image.width} height={image.height} className={classes.image} />
            {(isVideo(mediaItem) || isVideoFrame(mediaItem)) && <VideoFrame canvasRef={canvasRef} />}
        </>
    );
};

type ImageAnnotationsProps = {
    mediaItem: Media;
};

const ImageAnnotations = ({ mediaItem }: ImageAnnotationsProps) => {
    const { isFocussed } = useAnnotationVisibility();
    const { annotations } = useAnnotationActions();
    const { selectedAnnotations } = useSelectedAnnotations();

    // Order annotations by selection. Selected annotation should always be on top.
    const orderedAnnotations = [
        ...annotations.filter((a) => !selectedAnnotations.has(a.id)),
        ...annotations.filter((a) => selectedAnnotations.has(a.id)),
    ];

    const size = { width: mediaItem.width, height: mediaItem.height };

    return (
        <Annotations width={size.width} height={size.height} isFocussed={isFocussed} annotations={orderedAnnotations} />
    );
};

const PrefetchPredictions = () => {
    const { videoFrame } = useVideoPlayer();
    const nextFrameRangeIndexes = getVideoFrameRangeIndexes({
        frames: videoFrame.frame_count - 1,
        frameSkip: PREDICTION_FRAME_SKIP,
        frameNumber: videoFrame.frame_number,
        chunkSize: PREDICTION_CHUNK_SIZE,
    });
    usePrefetchVideoFramesPredictions({
        frameNumber: videoFrame.frame_number,
        frameSkip: PREDICTION_FRAME_SKIP,
        chunkSize: PREDICTION_CHUNK_SIZE,
    });
    usePrefetchVideoFramesPredictions({
        frameNumber: nextFrameRangeIndexes.endFrameIndex + 1,
        frameSkip: PREDICTION_FRAME_SKIP,
        chunkSize: PREDICTION_CHUNK_SIZE,
    });

    return null;
};

type MediaAnnotationsProps = {
    mediaItem: Media;
    mode: AnnotatorMode;
};

const MediaAnnotations = ({ mediaItem, mode }: MediaAnnotationsProps) => {
    const videoPlayerContext = useVideoPlayerContext();

    if (isVideoFrame(mediaItem) && videoPlayerContext?.videoControls?.isPlaying) {
        if (mode === 'annotation') {
            return <VideoAnnotations />;
        } else if (mode === 'prediction') {
            return (
                <>
                    <VideoPredictions />
                    <PrefetchPredictions />
                </>
            );
        }
    }

    return (
        <>
            <ImageAnnotations mediaItem={mediaItem} />
            {isVideoFrame(mediaItem) && mode === 'prediction' && <PrefetchPredictions />}
        </>
    );
};

type AnnotatorCanvasProps = {
    mediaItem: Media;
    image: ImageData;
    isReadOnly?: boolean;
    mode: AnnotatorMode;
};

export const AnnotatorCanvas = ({ mode, mediaItem, image, isReadOnly = false }: AnnotatorCanvasProps) => {
    const isSceneBusy = useIsAnnotatorSceneBusy();

    const areToolsDisabled = isSceneBusy || isReadOnly;

    const size = { width: mediaItem.width, height: mediaItem.height };

    return (
        <ZoomTransform target={size}>
            <div
                style={{ position: 'relative', height: '100%', width: '100%' }}
                onContextMenu={(event: MouseEvent): void => event.preventDefault()}
                className={isReadOnly ? classes.readOnlyCanvas : undefined}
            >
                <MediaImage image={image} mediaItem={mediaItem} />

                <MediaAnnotations mediaItem={mediaItem} mode={mode} />

                {!areToolsDisabled && <ToolManager />}
            </div>
        </ZoomTransform>
    );
};
