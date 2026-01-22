// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { UseQueryResult } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../api/client';
import type { ExtendedModel } from '../../../../constants/shared-types';

export const useGetModel = (modelId: string | null | undefined): UseQueryResult<ExtendedModel> => {
    const projectId = useProjectIdentifier();

    return $api.useQuery(
        'get',
        '/api/projects/{project_id}/models/{model_id}',
        { params: { path: { project_id: projectId, model_id: String(modelId) } } },
        { enabled: Boolean(modelId) }
    ) as UseQueryResult<ExtendedModel>;
};
