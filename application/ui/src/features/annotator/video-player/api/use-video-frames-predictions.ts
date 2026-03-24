// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { VideoFramePrediction } from '../../../../constants/shared-types';
import { mediaPredictionsQueryOptions } from '../../api/use-media-predictions';
import { usePredictionSetup } from '../../predictions-setup-provider.component';
import { useVideoPlayer } from '../video-player-provider.component';
import { getVideoFrameRangeIndexes } from './utils';

export const useVideoFramesPredictions = <T>({
    frameNumber,
    frameSkip,
    selector,
    rangeStride,
}: {
    frameNumber: number;
    frameSkip: number;
    rangeStride?: number;
    selector: (data: VideoFramePrediction[]) => T;
}) => {
    const projectId = useProjectIdentifier();
    const { videoFrame } = useVideoPlayer();
    const { selectedModelId } = usePredictionSetup();

    const { startFrameIndex, endFrameIndex } = getVideoFrameRangeIndexes({
        frames: videoFrame.frame_count - 1,
        frameSkip,
        frameNumber,
    });

    return useQuery({
        ...mediaPredictionsQueryOptions({
            projectId,
            modelId: selectedModelId,
            mediaId: videoFrame.id,
            range: { stride: rangeStride ?? frameSkip, start_frame: startFrameIndex, end_frame: endFrameIndex },
        }),
        select: selector,
    });
};
