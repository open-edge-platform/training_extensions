// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { $api } from 'src/api/client';
import { components } from 'src/api/openapi-spec';

type ServerLabel = components['schemas']['Label'];

export const useProjectLabels = (): ServerLabel[] => {
    const projectId = useProjectIdentifier();
    const { data: project } = $api.useSuspenseQuery('get', '/api/projects/{project_id}', {
        params: { path: { project_id: projectId } },
    });

    return project.task.labels || [];
};
