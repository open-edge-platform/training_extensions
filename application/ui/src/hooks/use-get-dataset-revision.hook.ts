// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '../api/client';
import { useProjectIdentifier } from './use-project-identifier.hook';

export const useGetDatasetRevision = (datasetRevisionId: string) => {
    const project_id = useProjectIdentifier();

    return $api.useQuery(
        'get',
        '/api/projects/{project_id}/dataset_revisions/{dataset_revision_id}',
        {
            params: {
                path: { project_id, dataset_revision_id: datasetRevisionId },
            },
        },
        {
            enabled: !!datasetRevisionId && datasetRevisionId !== 'unknown',
        }
    );
};
