// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useLocalStorage } from 'usehooks-ts';

import { useProjectIdentifier } from '../use-project-identifier.hook';
import { getParsedLocalStorage } from './utils';

const PREPARING_IMPORT_DATASET_KEY = (projectId: string) => `preparing-import-dataset-${projectId}`;

export const usePrepareImportDataset = () => {
    const projectId = useProjectIdentifier();

    const [_lsPreparingImportProject, setLsPreparingImportId] = useLocalStorage<string[]>(
        PREPARING_IMPORT_DATASET_KEY(projectId),
        () => getParsedLocalStorage(PREPARING_IMPORT_DATASET_KEY(projectId)) ?? []
    );

    const getLsPreparingImportIds = (): string[] => {
        return getParsedLocalStorage<string[]>(PREPARING_IMPORT_DATASET_KEY(projectId)) ?? [];
    };

    const addLsPreparingImportId = (jobId: string) => {
        return setLsPreparingImportId((prevState) => [...(prevState ?? []), jobId]);
    };

    const removeLsPreparingImportId = (jobId: string): void => {
        return setLsPreparingImportId((prevState) => prevState.filter((id) => id !== jobId));
    };

    return {
        addLsPreparingImportId,
        getLsPreparingImportIds,
        removeLsPreparingImportId,
    };
};
