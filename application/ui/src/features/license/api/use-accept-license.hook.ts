// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '../../../api/client';

export const useAcceptLicense = () => {
    return $api.useMutation('post', '/api/license/accept', {
        meta: { invalidateQueries: [['get', '/api/system/info']] },
    });
};
