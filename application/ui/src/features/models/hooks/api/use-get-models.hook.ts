// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { UseQueryResult } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../api/client';
import type { Model } from '../../../../constants/shared-types';

export const useGetModels = (): UseQueryResult<Model[]> => {
    const projectId = useProjectIdentifier();

    return $api.useSuspenseQuery('get', '/api/projects/{project_id}/models', {
        params: { path: { project_id: projectId } },
    }) as UseQueryResult<Model[]>;
};

export const useGetModelsQuery = () => {
    const projectId = useProjectIdentifier();

    return $api.useQuery('get', '/api/projects/{project_id}/models', {
        params: { path: { project_id: projectId } },
    });
};
