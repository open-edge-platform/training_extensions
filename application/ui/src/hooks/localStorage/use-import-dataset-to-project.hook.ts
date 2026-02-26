// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useLocalStorage } from 'usehooks-ts';

import { useProjectIdentifier } from '../use-project-identifier.hook';
import { getParsedLocalStorage } from './utils';

type DataValue = {
    size: number;
    fileName: string;
    prepareJobId: string | null;
    stagedDatasetId: string | null;
};

const IMPORT_DATASET_TO_PROJECT_KEY = (projectId: string) => `import-dataset-to-project-${projectId}`;

export const useImportDatasetToProject = () => {
    const projectId = useProjectIdentifier();

    const [lsImportDatasetToProject, setLsImportDatasetToProject] = useLocalStorage<DataValue[] | null>(
        IMPORT_DATASET_TO_PROJECT_KEY(projectId),
        () => getParsedLocalStorage<DataValue[]>(IMPORT_DATASET_TO_PROJECT_KEY(projectId)) ?? null
    );

    const getAllImportEntries = (): DataValue[] => {
        return lsImportDatasetToProject ?? [];
    };

    const findImportEntry = (item: Partial<DataValue>): DataValue | null => {
        return (
            lsImportDatasetToProject?.find(
                ({ prepareJobId, stagedDatasetId }) =>
                    item.prepareJobId === prepareJobId || item.stagedDatasetId === stagedDatasetId
            ) ?? null
        );
    };

    const appendImportEntry = (newEntry: DataValue) => {
        return setLsImportDatasetToProject((prev) => [...(prev ?? []), newEntry]);
    };

    const deleteImportEntry = (item: Partial<DataValue>): void => {
        return setLsImportDatasetToProject(
            (prev) =>
                prev?.filter(
                    ({ prepareJobId, stagedDatasetId }) =>
                        item.prepareJobId !== prepareJobId && item.stagedDatasetId !== stagedDatasetId
                ) ?? null
        );
    };

    const updateImportEntryStagedId = (prepareJobId: string, stagedDatasetId: string): void => {
        return setLsImportDatasetToProject(
            (prev) =>
                prev?.map((item) =>
                    item.prepareJobId === prepareJobId ? { ...item, prepareJobId: null, stagedDatasetId } : item
                ) ?? null
        );
    };

    const getLastImportEntry = (): DataValue | null => {
        return lsImportDatasetToProject ? lsImportDatasetToProject[lsImportDatasetToProject.length - 1] : null;
    };

    return {
        getAllImportEntries,
        appendImportEntry,
        findImportEntry,
        deleteImportEntry,
        updateImportEntryStagedId,
        getLastImportEntry,
    };
};
