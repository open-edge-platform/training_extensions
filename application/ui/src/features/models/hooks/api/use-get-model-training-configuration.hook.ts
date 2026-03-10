// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../api/client';

export const useGetModelTrainingConfiguration = (modelId: string | null) => {
    const projectId = useProjectIdentifier();

    return $api.useQuery(
        'get',
        '/api/projects/{project_id}/models/{model_id}/training_configuration',
        {
            params: { path: { project_id: projectId, model_id: modelId } },
        },
        {
            enabled: modelId !== null,
        }
    );
};

export const useGetModelArchitectureTrainingConfiguration = ({
    modelArchitectureId,
    modelRevisionId,
}: {
    modelArchitectureId: string | null;
    modelRevisionId: string | null;
}) => {
    const projectId = useProjectIdentifier();

    return $api.useQuery(
        'get',
        '/api/projects/{project_id}/training_configuration',
        {
            params: {
                path: {
                    project_id: projectId,
                },
                query: {
                    model_architecture_id: String(modelArchitectureId),
                },
            },
        },
        {
            enabled: modelArchitectureId !== null && modelRevisionId === null,
        }
    );
};
