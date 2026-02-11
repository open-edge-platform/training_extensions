// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useLocalStorage } from 'usehooks-ts';

import { useProjectIdentifier } from './use-project-identifier.hook';

const EXPORT_DATASET_KEY = (projectId: string) => `export-dataset-project-${projectId}`;

const getParsedLocalStorage = <T>(key: string): T | null => {
    if (Boolean(localStorage.getItem(key))) {
        return JSON.parse(localStorage.getItem(key) as string) as T;
    }

    return null;
};

export const useLocalStorageDataset = () => {
    const projectId = useProjectIdentifier();

    const [_lsExportProject, setLsExportId] = useLocalStorage<string[] | null>(
        EXPORT_DATASET_KEY(projectId),
        () => getParsedLocalStorage(EXPORT_DATASET_KEY(projectId)) ?? []
    );

    const getLsExportIds = (): string[] | null => {
        return getParsedLocalStorage<string[]>(EXPORT_DATASET_KEY(projectId));
    };

    const addLsExportId = (jobId: string) => {
        return setLsExportId((prevState) => [...(prevState ?? []), jobId]);
    };

    const removeLsExportId = (jobId: string): void => {
        return setLsExportId((prevState) => (prevState ? prevState.filter((id) => id !== jobId) : null));
    };

    return {
        getLsExportIds,
        addLsExportId,
        removeLsExportId,
    };
};
