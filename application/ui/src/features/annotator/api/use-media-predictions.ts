// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { queryOptions, useIsFetching, useQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { fetchClient } from '../../../api/client';
import { PredictionDTO, PredictionVideoRangePayload } from '../../../constants/shared-types';
import { EMPTY_LABEL_ID } from '../../../shared/annotator/labels';
import { getModelIdentifierPayload, SelectableModel } from '../../models/utils';

const MEDIA_PREDICTIONS_QUERY_KEY_PREFIX = (projectId: string, mediaId: string) => {
    return [projectId, 'media-predictions', mediaId];
};

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
            ...MEDIA_PREDICTIONS_QUERY_KEY_PREFIX(projectId, mediaId),
            selectedModel?.modelId,
            selectedModel?.id,
            range,
        ],
        queryFn: async () => {
            if (selectedModel === undefined) return [];

            const response = await fetchClient.POST('/api/projects/{project_id}/dataset/media/media:predict', {
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

export const useIsFetchingAnyPredictions = (mediaId: string) => {
    const projectId = useProjectIdentifier();

    const queryKey = MEDIA_PREDICTIONS_QUERY_KEY_PREFIX(projectId, mediaId);

    const numberOfFetchingPredictions = useIsFetching({ queryKey });

    return numberOfFetchingPredictions > 0;
};
