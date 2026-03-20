// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '../../../../api/client';

export const useUpdateTrainingConfigurationMutation = () => {
    return $api.useMutation('patch', '/api/projects/{project_id}/training_configuration');
};
