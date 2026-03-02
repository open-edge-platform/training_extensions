// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useLocalStorage } from 'usehooks-ts';

import { useProjectIdentifier } from '../use-project-identifier.hook';
import { getParsedLocalStorage } from './utils';

type DatasetImportState = {
    size: number;
    fileName: string;
    prepareJobId: string | null;
    stagedDatasetId: string | null;
    step: 'preparing' | 'labelMapping';
};

const IMPORT_DATASET_TO_PROJECT_KEY = (projectId: string) => `import-dataset-to-project-${projectId}`;

export const useImportDatasetToProject = () => {
    const projectId = useProjectIdentifier();

    const [lsImportDatasetToProject, setLsImportDatasetToProject] = useLocalStorage<DatasetImportState[] | null>(
        IMPORT_DATASET_TO_PROJECT_KEY(projectId),
        () => getParsedLocalStorage<DatasetImportState[]>(IMPORT_DATASET_TO_PROJECT_KEY(projectId)) ?? null
    );

    const getAllImportEntries = (): DatasetImportState[] => {
        return lsImportDatasetToProject ?? [];
    };

    const getImportEntry = (item: Partial<DatasetImportState>): DatasetImportState | null => {
        return (
            lsImportDatasetToProject?.find(
                ({ prepareJobId, stagedDatasetId }) =>
                    item.prepareJobId === prepareJobId || item.stagedDatasetId === stagedDatasetId
            ) ?? null
        );
    };

    const appendImportEntry = (newEntry: DatasetImportState) => {
        return setLsImportDatasetToProject((prev) => [...(prev ?? []), newEntry]);
    };

    const deleteImportEntry = (stagedDatasetId: string): void => {
        return setLsImportDatasetToProject(
            (prev) => prev?.filter((item) => item.stagedDatasetId !== stagedDatasetId) ?? null
        );
    };

    const updateImportEntryStep = (stagedDatasetId: string, newStep: 'preparing' | 'labelMapping'): void => {
        return setLsImportDatasetToProject(
            (prev) =>
                prev?.map((item) => (item.stagedDatasetId === stagedDatasetId ? { ...item, step: newStep } : item)) ??
                null
        );
    };

    const getLastImportEntry = (): DatasetImportState | null => {
        return lsImportDatasetToProject ? lsImportDatasetToProject[lsImportDatasetToProject.length - 1] : null;
    };

    return {
        getAllImportEntries,
        appendImportEntry,
        getImportEntry,
        deleteImportEntry,
        updateImportEntryStep,
        getLastImportEntry,
    };
};
