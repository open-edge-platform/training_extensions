// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getCurrentWindow } from '@tauri-apps/api/window';

import { EXPORT_DATASET_PREFIX } from '../hooks/storage/use-export-dataset.hook';
import { IMPORT_DATASET_AS_NEW_PROJECT_KEY } from '../hooks/storage/use-import-dataset-as-new-project.hook';
import { IMPORT_DATASET_TO_PROJECT_PREFIX } from '../hooks/storage/use-import-dataset-to-project.hook';

const DATASET_STORAGE_PREFIXES = [EXPORT_DATASET_PREFIX, IMPORT_DATASET_TO_PROJECT_PREFIX];
const DATASET_STORAGE_KEYS = [IMPORT_DATASET_AS_NEW_PROJECT_KEY];

const isDatasetStorageKey = (key: string): boolean =>
    DATASET_STORAGE_KEYS.includes(key) || DATASET_STORAGE_PREFIXES.some((prefix) => key.startsWith(prefix));

const cleanDatasetStorage = (): void => {
    Object.keys(localStorage)
        .filter(isDatasetStorageKey)
        .forEach((key) => localStorage.removeItem(key));
};

export const setupStorageCleanup = (): void => {
    void getCurrentWindow().onCloseRequested(cleanDatasetStorage);
};
