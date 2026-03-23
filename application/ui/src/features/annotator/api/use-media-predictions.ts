// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { queryOptions, useQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { fetchClient } from '../../../api/client';
import { Media, PredictionVideoRangePayload } from '../../../constants/shared-types';
import { useGetActiveModel } from '../../models/hooks/api/use-get-active-model.hook';

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
                    device: 'auto',
                    model_id: modelId,
                    save_predictions: false,
                    media: [{ media_id: mediaId, range }],
                },
            });

            if (response.error) return [];
            return response.data?.predictions ?? [];
        },
        enabled: modelId !== undefined,
    });

export const useMediaPredictions = ({ media }: { media: Media }) => {
    const projectId = useProjectIdentifier();
    const activeModel = useGetActiveModel();

    return useQuery(mediaPredictionsQueryOptions({ projectId, modelId: activeModel?.id, mediaId: media.id }));
};
