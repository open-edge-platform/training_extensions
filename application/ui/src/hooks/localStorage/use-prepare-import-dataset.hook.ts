// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useLocalStorage } from 'usehooks-ts';

import { useProjectIdentifier } from '../use-project-identifier.hook';
import { getParsedLocalStorage } from './utils';

type DataValue = {
    id: string;
    fileName: string;
};

const PREPARING_IMPORT_DATASET_KEY = (projectId: string) => `preparing-import-dataset-${projectId}`;

export const usePrepareImportDataset = () => {
    const projectId = useProjectIdentifier();

    const [lsPreparingImportProject, setLsPreparingImportId] = useLocalStorage<DataValue | null>(
        PREPARING_IMPORT_DATASET_KEY(projectId),
        () => getParsedLocalStorage<DataValue>(PREPARING_IMPORT_DATASET_KEY(projectId)) ?? null
    );

    const getLsPreparingImportId = (): DataValue | null => {
        return lsPreparingImportProject;
    };

    const addLsPreparingImportId = (jobId: string, fileName: string) => {
        return setLsPreparingImportId({ id: jobId, fileName });
    };

    const removeLsPreparingImportId = (): void => {
        return setLsPreparingImportId(null);
    };

    return {
        addLsPreparingImportId,
        getLsPreparingImportId,
        removeLsPreparingImportId,
    };
};
