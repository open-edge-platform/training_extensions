// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useSessionStorage } from 'usehooks-ts';

import { useProjectIdentifier } from '../use-project-identifier.hook';
import { getParsedSessionStorage } from './utils';

const EXPORT_DATASET_KEY = (projectId: string) => `export-dataset-${projectId}`;

type ExportDatasetData = {
    jobId: string;
    datasetId: string | null;
};

export const useExportDataset = () => {
    const projectId = useProjectIdentifier();

    const [_lsExportProject, setLsExportId] = useSessionStorage<ExportDatasetData[]>(
        EXPORT_DATASET_KEY(projectId),
        () => getParsedSessionStorage(EXPORT_DATASET_KEY(projectId)) ?? []
    );

    const getLsExportIds = (): ExportDatasetData[] => {
        return getParsedSessionStorage<ExportDatasetData[]>(EXPORT_DATASET_KEY(projectId)) ?? [];
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
