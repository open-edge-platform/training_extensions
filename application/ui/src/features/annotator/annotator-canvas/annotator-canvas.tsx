// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { MouseEvent, PointerEvent, useEffect, useRef, useState } from 'react';

import { Loading } from '@geti/ui';
import { useIsFetching } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { useSpinDelay } from 'spin-delay';

import { ZoomTransform } from '../../../components/zoom/zoom-transform';
import type { Media } from '../../../constants/shared-types';
import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';
import { useAnnotationVisibility } from '../../../shared/annotator/annotation-visibility-provider.component';
import type { AnnotatorMode } from '../../../shared/annotator/annotator-mode';
import { useAnnotator } from '../../../shared/annotator/annotator-provider.component';
import { useSelectedAnnotations } from '../../../shared/annotator/select-annotation-provider.component';
import { useTool } from '../../../shared/annotator/tool-provider.component';
import { useEditableAnnotationState } from '../../../shared/annotator/use-editable-annotation-state.hook';
import { isVideo, isVideoFrame } from '../../../shared/media-item-utils';
import { Annotations } from '../annotations/annotations.component';
import { VideoAnnotations, VideoPredictions } from '../annotations/video-annotations.component';
import { useIsAnnotatorSceneBusy } from '../hooks/use-is-annotator-scene-busy';
import { loadImageQueryOptions } from '../hooks/use-load-image-query.hook';
import { ToolManager } from '../tools/tool-manager.component';
import { usePrefetchVideoFramesAnnotations } from '../video-player/api/use-video-frames-annotations';
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

const PrefetchAnnotations = () => {
    const { videoFrame, step } = useVideoPlayer();

    const nextFrameRangeIndexes = getVideoFrameRangeIndexes({
        frames: videoFrame.frame_count - 1,
        frameSkip: step,
        frameNumber: videoFrame.frame_number,
    });

    usePrefetchVideoFramesAnnotations({ frameNumber: videoFrame.frame_number, frameSkip: step });
    usePrefetchVideoFramesAnnotations({ frameNumber: nextFrameRangeIndexes.endFrameIndex + 1, frameSkip: step });

    return null;
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
            return (
                <>
                    <VideoAnnotations />
                    <PrefetchAnnotations />
                </>
            );
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
            {isVideoFrame(mediaItem) && mode === 'annotation' && <PrefetchAnnotations />}
        </>
    );
};

type AnnotatorCanvasProps = {
    mediaItem: Media;
    image: ImageData;
    isReadOnly?: boolean;
    mode: AnnotatorMode;
};

type UseToolLayerPointerPassthroughProps = {
    canEditSelectedAnnotation: boolean;
};

const useToolLayerPointerPassthrough = ({ canEditSelectedAnnotation }: UseToolLayerPointerPassthroughProps) => {
    const { activeTool } = useTool();
    const isSelectionToolActive = activeTool === 'selection';
    const toolLayerRef = useRef<HTMLDivElement>(null);
    const [isToolLayerPointerPassthrough, setIsToolLayerPointerPassthrough] = useState(false);
    const pendingPointerRef = useRef<{ x: number; y: number } | null>(null);
    const animationFrameRef = useRef<number | null>(null);

    const evaluatePointerHit = () => {
        animationFrameRef.current = null;

        if (!canEditSelectedAnnotation || isSelectionToolActive) {
            setIsToolLayerPointerPassthrough(false);
            return;
        }

        const toolLayer = toolLayerRef.current;
        const pendingPointer = pendingPointerRef.current;

        if (toolLayer == null || pendingPointer == null) {
            return;
        }

        const previousPointerEvents = toolLayer.style.pointerEvents;
        toolLayer.style.pointerEvents = 'none';

        const hitElement = document.elementFromPoint(pendingPointer.x, pendingPointer.y);

        toolLayer.style.pointerEvents = previousPointerEvents;

        const shouldAllowAnnotationEdit =
            hitElement instanceof Element && hitElement.closest("[data-resize-anchor='true']") !== null;

        setIsToolLayerPointerPassthrough(shouldAllowAnnotationEdit);
    };

    const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
        if (!canEditSelectedAnnotation || isSelectionToolActive) {
            if (animationFrameRef.current !== null) {
                cancelAnimationFrame(animationFrameRef.current);
                animationFrameRef.current = null;
            }

            pendingPointerRef.current = null;
            setIsToolLayerPointerPassthrough(false);
            return;
        }

        pendingPointerRef.current = { x: event.clientX, y: event.clientY };

        if (animationFrameRef.current !== null) {
            return;
        }

        // Layers, order matters:
        // 1) Canvas layer: image/video.
        // 2) Annotations layer: annotation shapes and edit anchors.
        // 3) Tool layer: active drawing tool overlay.
        //
        // The tool layer sits above annotations and can consume pointer events while drawing tools are active.
        // To detect whether the pointer is over an editable anchor, we temporarily disable pointer events on the tool
        // layer, call `elementFromPoint`, then restore the previous pointer-events value.
        // We run this check at most once per animation frame to avoid doing DOM hit-testing on every pointer event.
        animationFrameRef.current = requestAnimationFrame(evaluatePointerHit);
    };

    useEffect(() => {
        return () => {
            if (animationFrameRef.current !== null) {
                cancelAnimationFrame(animationFrameRef.current);
            }
        };
    }, []);

    const toolLayerPointerEvents =
        isSelectionToolActive || (canEditSelectedAnnotation && isToolLayerPointerPassthrough) ? 'none' : 'auto';

    return { toolLayerRef, toolLayerPointerEvents, handlePointerMove } as const;
};

export const AnnotatorCanvas = ({ mode, mediaItem, image, isReadOnly = false }: AnnotatorCanvasProps) => {
    const projectId = useProjectIdentifier();
    const isSceneBusy = useIsAnnotatorSceneBusy();
    const { canvasRef } = useAnnotator();
    const { isSingleEditableSelection } = useEditableAnnotationState();
    const isFetchingMedia = useIsFetching({ queryKey: loadImageQueryOptions(projectId, mediaItem).queryKey }) > 0;

    const isLoadingMedia = useSpinDelay(isFetchingMedia, { delay: 300, minDuration: 200 });
    const areToolsDisabled = isSceneBusy || isReadOnly;
    const size = { width: mediaItem.width, height: mediaItem.height };
    const canEditSelectedAnnotation = !areToolsDisabled && isSingleEditableSelection;
    const { toolLayerRef, toolLayerPointerEvents, handlePointerMove } = useToolLayerPointerPassthrough({
        canEditSelectedAnnotation,
    });

    if (isLoadingMedia) {
        return <Loading size='M' />;
    }

    return (
        <ZoomTransform target={size}>
            <div
                style={{ position: 'relative', height: '100%', width: '100%' }}
                onContextMenu={(event: MouseEvent): void => event.preventDefault()}
                onPointerMove={handlePointerMove}
                className={isReadOnly ? classes.readOnlyCanvas : undefined}
                ref={canvasRef}
            >
                <MediaImage image={image} mediaItem={mediaItem} />
                <MediaAnnotations mediaItem={mediaItem} mode={mode} />

                {!areToolsDisabled && (
                    <div
                        ref={toolLayerRef}
                        style={{
                            position: 'absolute',
                            inset: 0,
                            pointerEvents: toolLayerPointerEvents,
                        }}
                    >
                        <ToolManager />
                    </div>
                )}
            </div>
        </ZoomTransform>
    );
};
