// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useImportJobStatus } from 'hooks/api/jobs/use-import-job-status.hook';
import { isJobFailed, isJobPending, isJobRunning } from 'hooks/api/util';

import { ImportActiveJob } from '../import-card-status/import-active-job/import-active-job.component';
import { ImportFailedJob } from '../import-card-status/import-failed-job/import-failed-job.component';

type PrepareImportDatasetProps = {
    size: number;
    jobId: string;
    fileName: string;
    stagedDatasetId: string;
    onSuccess: () => void;
    deleteEntry: () => void;
};

export const PrepareImportDataset = ({
    size,
    jobId,
    fileName,
    stagedDatasetId,
    onSuccess,
    deleteEntry,
}: PrepareImportDatasetProps) => {
    const { data: job, isError, error } = useImportJobStatus({ jobId, onSuccess });

    const isRunningOrPending = isJobRunning(job) || isJobPending(job);

    return (
        <>
            {isJobFailed(job) && (
                <ImportFailedJob
                    size={size}
                    fileName={fileName}
                    error={job.error ?? ''}
                    message={job.message ?? ''}
                    stagedDatasetId={stagedDatasetId}
                    deleteEntry={deleteEntry}
                />
            )}

            {isError && (
                <ImportFailedJob
                    size={size}
                    fileName={fileName}
                    error={`${error?.detail ?? 'Unknown error'}`}
                    message={'An error occurred during import preparation.'}
                    stagedDatasetId={stagedDatasetId}
                    deleteEntry={deleteEntry}
                />
            )}

            {isRunningOrPending && (
                <ImportActiveJob
                    job={job}
                    size={size}
                    fileName={fileName}
                    stagedDatasetId={stagedDatasetId}
                    deleteEntry={deleteEntry}
                />
            )}
        </>
    );
};
