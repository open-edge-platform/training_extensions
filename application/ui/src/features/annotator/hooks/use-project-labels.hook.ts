// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { get } from 'lodash-es';
import { $api } from 'src/api/client';

import { Label } from '../types';

export const useProjectLabels = (): Label[] => {
    const projectId = useProjectIdentifier();
    const { data: project } = $api.useQuery('get', '/api/projects/{project_id}', {
        params: { path: { project_id: projectId } },
    });

    return get(project, 'task.labels', []);
};
