// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isNil } from 'lodash-es';
import { useLocalStorage } from 'usehooks-ts';

import { useProjectIdentifier } from './use-project-identifier.hook';

const EXPORT_DATASET_KEY = (projectId: string) => `export-dataset-project-${projectId}`;

const getParsedLocalStorage = <T>(key: string): T | null => {
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

export const useLocalStorageDataset = () => {
    const projectId = useProjectIdentifier();

    const [_lsExportProject, setLsExportId] = useLocalStorage<string[]>(
        EXPORT_DATASET_KEY(projectId),
        () => getParsedLocalStorage(EXPORT_DATASET_KEY(projectId)) ?? []
    );

    const getLsExportIds = (): string[] => {
        return getParsedLocalStorage<string[]>(EXPORT_DATASET_KEY(projectId)) ?? [];
    };

    const addLsExportId = (jobId: string) => {
        return setLsExportId((prevState) => [...(prevState ?? []), jobId]);
    };

    const removeLsExportId = (jobId: string): void => {
        return setLsExportId((prevState) => prevState.filter((id) => id !== jobId));
    };

    return {
        getLsExportIds,
        addLsExportId,
        removeLsExportId,
    };
};
