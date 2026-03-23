// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { VideoFramePrediction } from '../../../../constants/shared-types';
import { useGetActiveModel } from '../../../models/hooks/api/use-get-active-model.hook';
import { mediaPredictionsQueryOptions } from '../../api/use-media-predictions';
import { useVideoPlayer } from '../video-player-provider.component';
import { getVideoFrameRangeIndexes } from './utils';

export const useVideoFramesPredictions = <T>({
    frameNumber,
    frameSkip,
    selector,
}: {
    frameNumber: number;
    frameSkip: number;
    selector: (data: VideoFramePrediction[]) => T;
}) => {
    const projectId = useProjectIdentifier();
    const { videoFrame } = useVideoPlayer();
    const activeModel = useGetActiveModel();

    const { startFrameIndex, endFrameIndex } = getVideoFrameRangeIndexes({
        frames: videoFrame.frame_count - 1,
        frameSkip,
        frameNumber,
    });

    return useQuery({
        ...mediaPredictionsQueryOptions({
            projectId,
            modelId: activeModel?.id,
            mediaId: videoFrame.id,
            range: { stride: 1, start_frame: startFrameIndex, end_frame: endFrameIndex },
        }),
        select: selector,
    });
};
