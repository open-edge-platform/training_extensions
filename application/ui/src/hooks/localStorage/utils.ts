// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isNil } from 'lodash-es';

export const getParsedLocalStorage = <T>(key: string): T | null => {
    const item = localStorage.getItem(key);

    if (isNil(item)) {
        return null;
    }

    try {
        return JSON.parse(item) as T;
    } catch {
        localStorage.removeItem(key);
        return null;
    }
};
