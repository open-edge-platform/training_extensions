// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../api/client';

export const useGetModelTrainingConfiguration = (modelId: string) => {
    const projectId = useProjectIdentifier();

    return $api.useQuery('get', '/api/projects/{project_id}/models/{model_id}/training_configuration', {
        params: { path: { project_id: projectId, model_id: modelId } },
    });
};
