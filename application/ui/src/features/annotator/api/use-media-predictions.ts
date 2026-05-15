// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { queryOptions, useIsFetching, useQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { fetchClient } from '../../../api/client';
import { PredictionDTO, PredictionVideoRangePayload } from '../../../constants/shared-types';
import { EMPTY_LABEL_ID } from '../../../shared/annotator/labels';
import { isVideoFrame } from '../../../shared/media-item-utils';
import { getModelIdentifierPayload, SelectableModel } from '../../models/utils';
import { usePredictionSetup } from '../predictions-setup-provider.component';
import { useSelectedMediaItem } from '../selected-media-item-provider.component';
import { PREDICTION_CHUNK_SIZE, PREDICTION_FRAME_SKIP } from '../video-player/api/use-video-frames-predictions';
import { getVideoFrameRangeIndexes } from '../video-player/api/utils';
import { useVideoPlayerContext } from '../video-player/video-player-provider.component';

export const mediaPredictionsQueryOptions = ({
    projectId,
    selectedModel,
    mediaId,
    range = null,
}: {
    projectId: string;
    selectedModel: SelectableModel | undefined;
    mediaId: string;
    range?: PredictionVideoRangePayload | null;
}) =>
    queryOptions({
        queryKey: [
            projectId,
            'media-predictions',
            mediaId,
            selectedModel?.modelId,
            selectedModel?.modelVariantId,
            range,
        ],
        queryFn: async ({ signal }) => {
            if (selectedModel === undefined) return [];

            const response = await fetchClient.POST('/api/projects/{project_id}/dataset/media/media:predict', {
                signal,
                params: { path: { project_id: projectId } },
                body: {
                    ...getModelIdentifierPayload(selectedModel),
                    device: 'AUTO',
                    save_predictions: false,
                    media: [{ media_id: mediaId, range }],
                },
            });

            if (response.error) return [];

            const predictions = response.data?.predictions ?? [];

            return predictions.map((predictionItem) => {
                if ((predictionItem.prediction ?? []).length === 0) {
                    return {
                        ...predictionItem,
                        prediction: [
                            {
                                shape: { type: 'full_image' },
                                labels: [{ id: EMPTY_LABEL_ID }],
                                confidences: [1],
                            } satisfies PredictionDTO,
                        ],
                    };
                }

                return predictionItem;
            });
        },
        staleTime: 1000 * 60 * 5,
        enabled: selectedModel !== undefined,
    });

export const useMediaPredictions = ({
    mediaId,
    selectedModel,
    range,
}: {
    mediaId: string;
    selectedModel: SelectableModel | undefined;
    range?: PredictionVideoRangePayload | null;
}) => {
    const projectId = useProjectIdentifier();

    return useQuery(mediaPredictionsQueryOptions({ projectId, selectedModel, mediaId, range }));
};

export const useIsFetchingCurrentRangeFramesPredictions = (mediaId: string) => {
    const projectId = useProjectIdentifier();

    const { selectedModel } = usePredictionSetup();
    const videoContext = useVideoPlayerContext();

    const frameNumber = videoContext?.videoFrame.frame_number ?? 0;
    const frameCount = videoContext?.videoFrame.frame_count ?? 1;

    // Exact query key for the range chunk covering the current frame (video only)
    const { startFrameIndex, endFrameIndex } = getVideoFrameRangeIndexes({
        frames: frameCount - 1,
        frameSkip: PREDICTION_FRAME_SKIP,
        frameNumber,
        chunkSize: PREDICTION_CHUNK_SIZE,
    });

    const rangeQueryKey = mediaPredictionsQueryOptions({
        projectId,
        selectedModel,
        mediaId,
        range: { stride: PREDICTION_FRAME_SKIP, start_frame: startFrameIndex, end_frame: endFrameIndex },
    }).queryKey;

    return useIsFetching({ queryKey: rangeQueryKey, exact: true }) > 0;
};

export const useIsFetchingCurrentFramePredictions = (mediaId: string) => {
    const projectId = useProjectIdentifier();
    const { selectedModel } = usePredictionSetup();
    const { mediaItem } = useSelectedMediaItem();

    const singleFrameRange = isVideoFrame(mediaItem)
        ? { start_frame: mediaItem.frame_number, end_frame: mediaItem.frame_number, stride: mediaItem.frame_stride }
        : null;

    const singleFrameQueryKey = mediaPredictionsQueryOptions({
        projectId,
        selectedModel,
        mediaId,
        range: singleFrameRange,
    }).queryKey;

    return useIsFetching({ queryKey: singleFrameQueryKey, exact: true }) > 0;
};

export const useIsFetchingPredictions = (mediaId: string) => {
    const videoContext = useVideoPlayerContext();

    const isPlaying = videoContext?.videoControls.isPlaying ?? false;

    const isFetchingRange = useIsFetchingCurrentRangeFramesPredictions(mediaId);
    const isFetchingSingleFrame = useIsFetchingCurrentFramePredictions(mediaId);

    return isPlaying ? isFetchingRange : isFetchingSingleFrame;
};
