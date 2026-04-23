// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { usePrefetchQuery, useQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../api/client';
import { AnnotatedVideoFrame } from '../../../../constants/shared-types';
import { useVideoPlayer } from '../video-player-provider.component';
import { getVideoFrameRangeIndexes } from './utils';

const useGetVideoFramesAnnotationsQueryOptions = ({
    frameNumber,
    frameSkip,
}: {
    frameSkip: number;
    frameNumber: number;
}) => {
    const projectId = useProjectIdentifier();
    const { videoFrame } = useVideoPlayer();

    const frames = videoFrame.frame_count - 1;

    const { startFrameIndex, endFrameIndex } = getVideoFrameRangeIndexes({ frames, frameSkip, frameNumber });

    return $api.queryOptions('get', '/api/projects/{project_id}/dataset/media/{media_id}/frames', {
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
    });
};

export const usePrefetchVideoFramesAnnotations = ({
    frameNumber,
    frameSkip,
}: {
    frameNumber: number;
    frameSkip: number;
}) => {
    const queryOptions = useGetVideoFramesAnnotationsQueryOptions({ frameNumber, frameSkip });

    return usePrefetchQuery(queryOptions);
};

export const useVideoFramesAnnotations = <T>({
    frameNumber,
    frameSkip,
    selector,
}: {
    frameNumber: number;
    frameSkip: number;
    selector: (data: AnnotatedVideoFrame[]) => T;
}) => {
    const queryOptions = useGetVideoFramesAnnotationsQueryOptions({
        frameSkip,
        frameNumber,
    });

    return useQuery({
        ...queryOptions,
        select: selector,
        // We invalidate cache manually when the user annotates a frame, so we can set an infinite stale time to
        // avoid unnecessary refetches
        staleTime: Infinity,
    });
};
