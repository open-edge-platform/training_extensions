// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useLocalStorage } from 'usehooks-ts';

import { ImportDatasetToProjectState } from '../../features/dataset/import-export/import-dataset/util';
import { useProjectIdentifier } from '../use-project-identifier.hook';
import { DatasetImportState, getParsedLocalStorage } from './utils';

export type DatasetImportToProjectState = DatasetImportState<ImportDatasetToProjectState>;

const IMPORT_DATASET_TO_PROJECT_KEY = (projectId: string) => `import-dataset-to-project-${projectId}`;

export const useImportDatasetToProject = () => {
    const projectId = useProjectIdentifier();

    const [lsImportDatasetToProject, setLsImportDatasetToProject] = useLocalStorage<
        DatasetImportToProjectState[] | null
    >(
        IMPORT_DATASET_TO_PROJECT_KEY(projectId),
        () => getParsedLocalStorage<DatasetImportToProjectState[]>(IMPORT_DATASET_TO_PROJECT_KEY(projectId)) ?? null
    );

    const getAllImportEntries = (): DatasetImportToProjectState[] => {
        return lsImportDatasetToProject ?? [];
    };

    const getImportEntry = (stagedDatasetId: string): DatasetImportToProjectState | null => {
        return lsImportDatasetToProject?.find((item) => item.stagedDatasetId === stagedDatasetId) ?? null;
    };

    const appendImportEntry = (newEntry: DatasetImportToProjectState) => {
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

    const updateImportEntry = (stagedDatasetId: string, newImportState: Partial<DatasetImportToProjectState>): void => {
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
