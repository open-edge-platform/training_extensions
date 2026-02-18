// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useLocalStorage } from 'usehooks-ts';

import { useProjectIdentifier } from '../use-project-identifier.hook';
import { getParsedLocalStorage } from './utils';

const EXPORT_DATASET_KEY = (projectId: string) => `export-dataset-${projectId}`;

type exportDatasetData = {
    jobId: string;
    datasetId: string | null;
};

export const useExportDataset = () => {
    const projectId = useProjectIdentifier();

    const [_lsExportProject, setLsExportId] = useLocalStorage<exportDatasetData[]>(
        EXPORT_DATASET_KEY(projectId),
        () => getParsedLocalStorage(EXPORT_DATASET_KEY(projectId)) ?? []
    );

    const getLsExportIds = (): exportDatasetData[] => {
        return getParsedLocalStorage<exportDatasetData[]>(EXPORT_DATASET_KEY(projectId)) ?? [];
    };

    const addLsExportId = (jobId: string, datasetId: string | null) => {
        return setLsExportId((prevState) => [...(prevState ?? []), { jobId, datasetId }]);
    };

    const removeLsExportId = (jobId: string): void => {
        return setLsExportId((prevState) => prevState.filter((id) => id.jobId !== jobId));
    };

    return {
        getLsExportIds,
        addLsExportId,
        removeLsExportId,
    };
};
