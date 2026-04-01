// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { usePrefetchQuery, useSuspenseQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../api/client';
import { isSuccessfulModel } from '../../model-listing/utils/utils';

const getModelsQueryOptions = (projectId: string) => {
    return $api.queryOptions('get', '/api/projects/{project_id}/models', {
        params: { path: { project_id: projectId } },
    });
};

export const useGetModels = () => {
    const projectId = useProjectIdentifier();

    return useSuspenseQuery(getModelsQueryOptions(projectId));
};

export const useGetSuccessfulModels = () => {
    const projectId = useProjectIdentifier();

    return useSuspenseQuery({
        ...getModelsQueryOptions(projectId),
        select: (models) => models.filter((model) => isSuccessfulModel(model) && !model.files_deleted),
    });
};

export const usePrefetchModels = () => {
    const projectId = useProjectIdentifier();

    return usePrefetchQuery(getModelsQueryOptions(projectId));
};
