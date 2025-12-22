// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from 'src/api/client';

export const useUpdateLabel = () => {
    return $api.useMutation('patch', '/api/projects/{project_id}/labels');
};
