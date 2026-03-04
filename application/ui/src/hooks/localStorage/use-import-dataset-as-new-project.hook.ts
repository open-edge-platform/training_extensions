// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useLocalStorage } from 'usehooks-ts';

import { ImportDatasetAsNewProjectState } from '../../features/dataset/import-export/import-dataset/util';
import { useProjectIdentifier } from '../use-project-identifier.hook';
import { getParsedLocalStorage } from './utils';

type DatasetImportState = {
    size: number;
    fileName: string;
    importJobId: string | null;
    prepareJobId: string | null;
    stagedDatasetId: string | null;
    step: ImportDatasetAsNewProjectState;
};

const IMPORT_DATASET_AS_NEW_PROJECT_KEY = (projectId: string) => `import-dataset-as-new-project-${projectId}`;

export const useImportDatasetAsNewProject = () => {
    const projectId = useProjectIdentifier();

    const [lsImportDatasetAsNewProject, setLsImportDatasetAsNewProject] = useLocalStorage<DatasetImportState[] | null>(
        IMPORT_DATASET_AS_NEW_PROJECT_KEY(projectId),
        () => getParsedLocalStorage<DatasetImportState[]>(IMPORT_DATASET_AS_NEW_PROJECT_KEY(projectId)) ?? null
    );

    const getAllImportEntries = (): DatasetImportState[] => {
        return lsImportDatasetAsNewProject ?? [];
    };

    const getImportEntry = (stagedDatasetId: string): DatasetImportState | null => {
        return lsImportDatasetAsNewProject?.find((item) => item.stagedDatasetId === stagedDatasetId) ?? null;
    };

    const appendImportEntry = (newEntry: DatasetImportState) => {
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

    const updateImportEntry = (stagedDatasetId: string, newImportState: Partial<DatasetImportState>): void => {
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
