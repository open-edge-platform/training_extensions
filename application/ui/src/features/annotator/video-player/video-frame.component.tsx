// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { RefObject, useEffect, useRef } from 'react';

import { VisuallyHidden } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { type Media } from '../../../constants/shared-types';
import { getMediaBinaryUrl } from '../../../shared/media-url.utils';

type VideoFrameProps = {
    canvasRef: RefObject<HTMLCanvasElement | null>;
    mediaItem: Media;
};

const useRequestVideoFrameCallback = (canvasRef: RefObject<HTMLCanvasElement | null>) => {
    const videoRef = useRef<HTMLVideoElement>(null);

    useEffect(() => {
        if (videoRef.current === null) {
            return;
        }

        const video = videoRef.current;

        let callbackId: number | null = null;

        const updateCanvas: VideoFrameRequestCallback = (now, metadata) => {
            if (videoRef.current === null) {
                return;
            }

            const ctx = canvasRef.current?.getContext('2d');

            if (ctx == null) {
                return;
            }

            console.log({ now, metadata });

            ctx.drawImage(video, 0, 0);

            callbackId = video.requestVideoFrameCallback(updateCanvas);
        };

        callbackId = video.requestVideoFrameCallback(updateCanvas);

        return () => {
            if (callbackId !== null) {
                video.cancelVideoFrameCallback(callbackId);
            }
        };
    }, [canvasRef]);

    useEffect(() => {
        if (videoRef.current === null) {
            return;
        }

        // videoRef.current.play();
    }, []);

    return videoRef;
};

export const VideoFrame = ({ canvasRef, mediaItem }: VideoFrameProps) => {
    const projectId = useProjectIdentifier();
    const videoRef = useRequestVideoFrameCallback(canvasRef);

    return (
        <VisuallyHidden>
            <video
                ref={videoRef}
                src={getMediaBinaryUrl(projectId, mediaItem.id)}
                width={mediaItem.width}
                height={mediaItem.height}
                preload={'auto'}
                muted
            />
        </VisuallyHidden>
    );
};
