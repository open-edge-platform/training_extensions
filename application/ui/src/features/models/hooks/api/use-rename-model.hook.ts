// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '../../../../api/client';

export const useRenameModel = () => {
    return $api.useMutation('patch', '/api/projects/{project_id}/models/{model_id}', {
        meta: {
            invalidateQueries: [['get', '/api/projects/{project_id}/models']],
        },
    });
};
