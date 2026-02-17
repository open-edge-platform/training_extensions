// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '../../../../../api/client';
import { ExportDatasetJob } from '../../../../../constants/shared-types';
import { useExportDataset } from '../../../../../hooks/localStorage/use-export-dataset.hook';
import { isInvalidJob, isJobDone, isJobFailed } from '../util';

export const useExportStatus = (jobId: string) => {
    const { removeLsExportId } = useExportDataset();

    const { data, ...others } = $api.useQuery(
        'get',
        '/api/jobs/{job_id}',
        { params: { path: { job_id: jobId } } },
        {
            refetchInterval: ({ state }) => (isJobDone(state.data) || isJobFailed(state.data) ? false : 1_000),
            select: (currentData) => currentData as ExportDatasetJob,
        }
    );

    if (others.isError && isInvalidJob(others.error)) {
        removeLsExportId(jobId);
    }

    return {
        job: data,
        ...others,
    };
};
