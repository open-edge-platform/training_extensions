// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '../../../../api/client';

export const useGetModelArchitectures = () => {
    return $api.useSuspenseQuery('get', '/api/model_architectures');
};
