// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { isInvalidJob, isJobDone, isJobFailed } from 'hooks/api/util';

import { $api } from '../../../../../api/client';
import { ExportDatasetJob } from '../../../../../constants/shared-types';
import { useExportDataset } from '../../../../../hooks/storage/use-export-dataset.hook';

export const useExportStatus = (jobId: string) => {
    const { removeLsExportId } = useExportDataset();

    const response = $api.useQuery(
        'get',
        '/api/jobs/{job_id}',
        { params: { path: { job_id: jobId } } },
        {
            refetchInterval: ({ state }) => {
                return isJobDone(state.data) || isJobFailed(state.data) || state.status === 'error' ? false : 1_000;
            },
            select: (currentData) => currentData as ExportDatasetJob,
        }
    );

    useEffect(() => {
        if (response.isError && isInvalidJob(response.error)) {
            removeLsExportId(jobId);
        }
    }, [jobId, removeLsExportId, response.error, response.isError]);

    return response;
};
