// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '../../../../api/client';

export const useGetTrainingDevices = () => {
    return $api.useSuspenseQuery('get', '/api/system/devices/training');
};
