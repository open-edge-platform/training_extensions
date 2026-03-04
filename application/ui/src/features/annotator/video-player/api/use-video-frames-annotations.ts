// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../api/client';
import { AnnotatedVideoFrame } from '../../../../constants/shared-types';
import { useVideoPlayer } from '../video-player-provider.component';

const CHUNK_SIZE = 30;

export const useVideoFramesAnnotations = <T>({
    frameNumber,
    frameSkip,
    selector,
}: {
    frameNumber: number;
    frameSkip: number;
    selector: (data: AnnotatedVideoFrame[]) => T;
}) => {
    const projectId = useProjectIdentifier();
    const { videoFrame } = useVideoPlayer();

    const annotationChunkSize = CHUNK_SIZE * frameSkip;

    const frames = videoFrame.frame_count - 1;

    const startFrameIndex = Math.floor(frameNumber / annotationChunkSize) * annotationChunkSize;
    const endFrameIndex = Math.min(startFrameIndex + annotationChunkSize - 1, frames);

    return $api.useQuery(
        'get',
        '/api/projects/{project_id}/dataset/media/{media_id}/frames',
        {
            params: {
                path: {
                    project_id: projectId,
                    media_id: videoFrame.id,
                },
                query: {
                    frame_index_from: startFrameIndex,
                    frame_index_to: endFrameIndex,
                },
            },
        },
        {
            select: selector,
            // We invalidate cache manually when the user annotates a frame, so we can set infinite cache and stale
            // time to avoid unnecessary refetches
            staleTime: Infinity,
        }
    );
};
