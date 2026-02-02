// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '../../../../api/client';

export const useRenameDatasetRevision = () => {
    return $api.useMutation('patch', '/api/projects/{project_id}/dataset_revisions/{dataset_revision_id}', {
        meta: {
            invalidateQueries: [['get', '/api/projects/{project_id}/dataset_revisions']],
        },
    });
};
