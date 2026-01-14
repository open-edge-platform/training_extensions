// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../api/client';

export const useGetModel = (modelId: string | null | undefined) => {
    const projectId = useProjectIdentifier();

    if (!modelId) {
        return undefined;
    }

    return $api.useSuspenseQuery('get', '/api/projects/{project_id}/models/{model_id}', {
        params: { path: { project_id: projectId, model_id: modelId } },
    });
};
