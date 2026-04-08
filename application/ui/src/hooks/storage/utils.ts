// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isNil } from 'lodash-es';

export type DatasetImportState<T> = {
    size: number;
    fileName: string;
    importJobId: string | null;
    prepareJobId: string;
    stagedDatasetId: string;
    step: T;
};

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

export const getParsedSessionStorage = <T>(key: string): T | null => {
    const item = sessionStorage.getItem(key);

    if (isNil(item)) {
        return null;
    }

    try {
        return JSON.parse(item) as T;
    } catch {
        sessionStorage.removeItem(key);
        return null;
    }
};
