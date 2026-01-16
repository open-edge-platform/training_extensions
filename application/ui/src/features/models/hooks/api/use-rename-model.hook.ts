// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '../../../../api/client';

export const useRenameModel = (modelId: string | null | undefined) => {
    if (!modelId) {
        return undefined;
    }

    return $api.useMutation('patch', '/api/projects/{project_id}/models/{model_id}', {
        meta: {
            invalidateQueries: [['get', '/api/projects/{project_id}/models']],
        },
    });
};
