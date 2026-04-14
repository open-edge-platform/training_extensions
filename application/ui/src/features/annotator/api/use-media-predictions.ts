// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { queryOptions, useQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { fetchClient } from '../../../api/client';
import { PredictionDTO, PredictionVideoRangePayload } from '../../../constants/shared-types';
import { EMPTY_LABEL_ID } from '../../../shared/annotator/labels';

export const mediaPredictionsQueryOptions = ({
    projectId,
    modelId,
    mediaId,
    range = null,
}: {
    projectId: string;
    modelId: string | undefined;
    mediaId: string;
    range?: PredictionVideoRangePayload | null;
}) =>
    queryOptions({
        queryKey: [projectId, 'media-predictions', mediaId, modelId, range],
        queryFn: async () => {
            if (modelId === undefined) return [];

            const response = await fetchClient.POST('/api/projects/{project_id}/dataset/media/media:predict', {
                params: { path: { project_id: projectId } },
                body: {
                    device: 'AUTO',
                    model_id: modelId,
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
        enabled: modelId !== undefined,
    });

export const useMediaPredictions = ({
    mediaId,
    modelId,
    range,
}: {
    mediaId: string;
    modelId: string | undefined;
    range?: PredictionVideoRangePayload | null;
}) => {
    const projectId = useProjectIdentifier();

    return useQuery(mediaPredictionsQueryOptions({ projectId, modelId, mediaId, range }));
};
