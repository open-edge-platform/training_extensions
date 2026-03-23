// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api, fetchClient } from '../../../../api/client';
import { AnnotatedVideoFrame } from '../../../../constants/shared-types';
import { useVideoPlayer } from '../video-player-provider.component';

const getVideoFramesPredictionsQueryOptions = async () =>
    $api.queryOptions('post', '/api/projects/{project_id}/dataset/media/media:predict');

export const useVideoFramesPredictions = <T>({
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
    useQuery({
        queryKey: ['video-frames-predictions'],
        queryFn: async () => {
            const response = await fetchClient.POST('/api/projects/{project_id}/dataset/media/media:predict', {
                params: {
                    path: {
                        project_id: projectId,
                    },
                },
                body: {
                    device: 'cpu',
                },
            });
        },
    });
};
