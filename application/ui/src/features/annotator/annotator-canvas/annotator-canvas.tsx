// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { MouseEvent, useEffect, useRef } from 'react';

import { ZoomTransform } from '../../../components/zoom/zoom-transform';
import type { Media } from '../../../constants/shared-types';
import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';
import { useAnnotationVisibility } from '../../../shared/annotator/annotation-visibility-provider.component';
import { useSelectedAnnotations } from '../../../shared/annotator/select-annotation-provider.component';
import { isVideo, isVideoFrame } from '../../../shared/media-item-utils';
import { Annotations } from '../annotations/annotations.component';
import { useMediaItemImage } from '../selected-media-item-provider.component';
import { ToolManager } from '../tools/tool-manager.component';
import { useIsAnnotatorSceneBusy } from '../use-is-annotator-scene-busy';
import { VideoFrame } from '../video-player/video-frame.component';

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
            {(isVideo(mediaItem) || isVideoFrame(mediaItem)) && (
                <VideoFrame canvasRef={canvasRef} mediaItem={mediaItem} />
            )}
        </>
    );
};

type AnnotatorCanvasProps = {
    mediaItem: Media;
    isReadOnly?: boolean;
};

export const AnnotatorCanvas = ({ mediaItem, isReadOnly = false }: AnnotatorCanvasProps) => {
    const { annotations } = useAnnotationActions();
    const { selectedAnnotations } = useSelectedAnnotations();
    const { isFocussed } = useAnnotationVisibility();
    const { image } = useMediaItemImage();
    const isSceneBusy = useIsAnnotatorSceneBusy();

    const areToolsDisabled = isSceneBusy || isReadOnly;

    // Order annotations by selection. Selected annotation should always be on top.
    const orderedAnnotations = [
        ...annotations.filter((a) => !selectedAnnotations.has(a.id)),
        ...annotations.filter((a) => selectedAnnotations.has(a.id)),
    ];

    const size = { width: mediaItem.width, height: mediaItem.height };

    return (
        <ZoomTransform target={size}>
            <div
                style={{ position: 'relative', height: '100%', width: '100%' }}
                onContextMenu={(event: MouseEvent): void => event.preventDefault()}
            >
                <MediaImage image={image} mediaItem={mediaItem} />

                <Annotations
                    width={size.width}
                    height={size.height}
                    isFocussed={isFocussed}
                    annotations={orderedAnnotations}
                />
                {!areToolsDisabled && <ToolManager />}
            </div>
        </ZoomTransform>
    );
};
