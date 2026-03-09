// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useLocalStorage } from 'usehooks-ts';

import { ImportDatasetToProjectState } from '../../features/dataset/import-export/import-dataset/util';
import { useProjectIdentifier } from '../use-project-identifier.hook';
import { getParsedLocalStorage } from './utils';

type DatasetImportState = {
    size: number;
    fileName: string;
    importJobId: string | null;
    prepareJobId: string | null;
    stagedDatasetId: string | null;
    step: ImportDatasetToProjectState;
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

    const getImportEntry = (stagedDatasetId: string): DatasetImportState | null => {
        return lsImportDatasetToProject?.find((item) => item.stagedDatasetId === stagedDatasetId) ?? null;
    };

    const appendImportEntry = (newEntry: DatasetImportState) => {
        return setLsImportDatasetToProject((prev) => [...(prev ?? []), newEntry]);
    };

    const deleteImportEntry = (stagedDatasetId: string): void => {
        return setLsImportDatasetToProject(
            (prev) => prev?.filter((item) => item.stagedDatasetId !== stagedDatasetId) ?? null
        );
    };

    const updateImportEntryStep = (stagedDatasetId: string, newStep: ImportDatasetToProjectState): void => {
        return setLsImportDatasetToProject(
            (prev) =>
                prev?.map((item) => (item.stagedDatasetId === stagedDatasetId ? { ...item, step: newStep } : item)) ??
                null
        );
    };

    const updateImportEntry = (stagedDatasetId: string, newImportState: Partial<DatasetImportState>): void => {
        return setLsImportDatasetToProject(
            (prev) =>
                prev?.map((item) =>
                    item.stagedDatasetId === stagedDatasetId ? { ...item, ...newImportState } : item
                ) ?? []
        );
    };

    return {
        getAllImportEntries,
        appendImportEntry,
        getImportEntry,
        deleteImportEntry,
        updateImportEntryStep,
        updateImportEntry,
    };
};
