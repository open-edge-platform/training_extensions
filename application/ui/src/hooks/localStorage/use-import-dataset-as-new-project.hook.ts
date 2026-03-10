// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useLocalStorage } from 'usehooks-ts';

import { ImportDatasetAsNewProjectState } from '../../features/dataset/import-export/import-dataset/util';
import { DatasetImportState, getParsedLocalStorage } from './utils';

export type DatasetImportAsNewProjectState = DatasetImportState<ImportDatasetAsNewProjectState>;

const IMPORT_DATASET_AS_NEW_PROJECT_KEY = `import-dataset-as-new-project`;

export const useImportDatasetAsNewProject = () => {
    const [lsImportDatasetAsNewProject, setLsImportDatasetAsNewProject] = useLocalStorage<
        DatasetImportAsNewProjectState[] | null
    >(
        IMPORT_DATASET_AS_NEW_PROJECT_KEY,
        () => getParsedLocalStorage<DatasetImportAsNewProjectState[]>(IMPORT_DATASET_AS_NEW_PROJECT_KEY) ?? null
    );

    const getAllImportEntries = (): DatasetImportAsNewProjectState[] => {
        return lsImportDatasetAsNewProject ?? [];
    };

    const getImportEntry = (stagedDatasetId: string): DatasetImportAsNewProjectState | null => {
        return lsImportDatasetAsNewProject?.find((item) => item.stagedDatasetId === stagedDatasetId) ?? null;
    };

    const appendImportEntry = (newEntry: DatasetImportAsNewProjectState) => {
        return setLsImportDatasetAsNewProject((prev) => [...(prev ?? []), newEntry]);
    };

    const deleteImportEntry = (stagedDatasetId: string): void => {
        return setLsImportDatasetAsNewProject(
            (prev) => prev?.filter((item) => item.stagedDatasetId !== stagedDatasetId) ?? null
        );
    };

    const updateImportEntryStep = (stagedDatasetId: string, newStep: ImportDatasetAsNewProjectState): void => {
        return setLsImportDatasetAsNewProject(
            (prev) =>
                prev?.map((item) => (item.stagedDatasetId === stagedDatasetId ? { ...item, step: newStep } : item)) ??
                null
        );
    };

    const updateImportEntry = (
        stagedDatasetId: string,
        newImportState: Partial<DatasetImportAsNewProjectState>
    ): void => {
        return setLsImportDatasetAsNewProject(
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
