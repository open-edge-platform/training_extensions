// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../api/client';

export const useGetModels = () => {
    const projectId = useProjectIdentifier();

    return $api.useSuspenseQuery('get', '/api/projects/{project_id}/models', {
        params: { path: { project_id: projectId } },
    });
};
