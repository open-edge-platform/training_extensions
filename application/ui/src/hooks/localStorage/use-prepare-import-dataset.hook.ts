// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useLocalStorage } from 'usehooks-ts';

import { useProjectIdentifier } from '../use-project-identifier.hook';
import { getParsedLocalStorage } from './utils';

type DataValue = {
    id: string;
    size: number;
    fileName: string;
};

const PREPARING_IMPORT_DATASET_KEY = (projectId: string) => `preparing-import-dataset-${projectId}`;

export const usePrepareImportDataset = () => {
    const projectId = useProjectIdentifier();

    const [lsPreparingImportDataset, setLsPreparingImport] = useLocalStorage<DataValue | null>(
        PREPARING_IMPORT_DATASET_KEY(projectId),
        () => getParsedLocalStorage<DataValue>(PREPARING_IMPORT_DATASET_KEY(projectId)) ?? null
    );
    const getLsPreparingImport = (): DataValue | null => {
        return lsPreparingImportDataset;
    };

    const addLsPreparingImport = (jobId: string, fileName: string, size: number) => {
        return setLsPreparingImport({ id: jobId, fileName, size });
    };

    const removeLsPreparingImport = (): void => {
        return setLsPreparingImport(null);
    };

    return {
        addLsPreparingImport,
        getLsPreparingImport,
        removeLsPreparingImport,
    };
};
