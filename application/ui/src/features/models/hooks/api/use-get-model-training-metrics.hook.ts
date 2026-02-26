// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../api/client';

export const useGetModelTrainingMetrics = (modelId: string | null | undefined) => {
    const projectId = useProjectIdentifier();

    return $api.useSuspenseQuery('get', '/api/projects/{project_id}/models/{model_id}/training_metrics', {
        params: { path: { project_id: projectId, model_id: String(modelId) } },
    });
};
