// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '../../../../api/client';

export const useDeleteModel = () => {
    return $api.useMutation('delete', '/api/projects/{project_id}/models/{model_id}', {
        meta: {
            invalidateQueries: [['get', '/api/projects/{project_id}/models']],
        },
    });
};
