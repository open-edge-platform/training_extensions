// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { EXPORT_DATASET_PREFIX } from './use-export-dataset.hook';
import { IMPORT_DATASET_AS_NEW_PROJECT_KEY } from './use-import-dataset-as-new-project.hook';
import { IMPORT_DATASET_TO_PROJECT_PREFIX } from './use-import-dataset-to-project.hook';

const DATASET_STORAGE_PREFIXES = [EXPORT_DATASET_PREFIX, IMPORT_DATASET_TO_PROJECT_PREFIX];
const DATASET_STORAGE_KEYS = [IMPORT_DATASET_AS_NEW_PROJECT_KEY];

const isDatasetStorageKey = (key: string): boolean =>
    DATASET_STORAGE_KEYS.includes(key) || DATASET_STORAGE_PREFIXES.some((prefix) => key.startsWith(prefix));

const cleanDatasetStorage = (): void => {
    Object.keys(localStorage)
        .filter(isDatasetStorageKey)
        .forEach((key) => localStorage.removeItem(key));
};

const isTauri = (): boolean => '__TAURI_INTERNALS__' in window;

export const setupTauriStorageCleanup = (): void => {
    if (!isTauri()) {
        return;
    }

    import('@tauri-apps/api/window')
        .then(({ getCurrentWindow }) => getCurrentWindow().onCloseRequested(cleanDatasetStorage))
        .catch(console.error);
};
