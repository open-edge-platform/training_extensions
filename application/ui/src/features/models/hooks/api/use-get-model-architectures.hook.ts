// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProject } from 'hooks/api/project.hook';

import { $api } from '../../../../api/client';

export const useGetTaskModelArchitectures = () => {
    const { data: projectData } = useProject();

    return $api.useSuspenseQuery('get', '/api/model_architectures', {
        params: {
            query: {
                task: projectData.task.task_type,
            },
        },
    });
};
