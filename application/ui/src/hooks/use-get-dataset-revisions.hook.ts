// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '../api/client';
import { useProjectIdentifier } from './use-project-identifier.hook';

export const useGetDatasetRevisions = () => {
    const project_id = useProjectIdentifier();

    return $api.useQuery('get', '/api/projects/{project_id}/dataset_revisions', {
        params: {
            path: { project_id },
        },
    });
};
