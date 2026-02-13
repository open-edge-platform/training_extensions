// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '../../../../api/client';
import { ExportDatasetJob } from '../../../../constants/shared-types';
import { useExportDataset } from '../../../../hooks/localStorage/use-export-dataset.hook';
import { isInvalidJob, isJobDone, isJobFailed } from './util';

export const useExportStatus = (jobId: string) => {
    const { removeLsExportId } = useExportDataset();

    const jobResponse = $api.useQuery(
        'get',
        '/api/jobs/{job_id}',
        { params: { path: { job_id: jobId } } },
        {
            refetchInterval: ({ state }) => (isJobDone(state.data) || isJobFailed(state.data) ? false : 1_000),
            select: (data) => data as ExportDatasetJob,
        }
    );

    const datasetResponse = $api.useQuery('get', '/api/staged_datasets', undefined, {
        enabled: isJobDone(jobResponse.data),
    });

    if (jobResponse.isError && isInvalidJob(jobResponse.error)) {
        removeLsExportId(jobId);
    }

    return {
        job: jobResponse.data,
        zipItems: isJobDone(jobResponse.data) ? datasetResponse.data : null,
        isFetching: jobResponse.isFetching || datasetResponse.isFetching,
    };
};
