// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { View } from '@geti/ui';
import { useImportJobStatus } from 'hooks/api/jobs/use-import-job-status.hook';
import { isJobDone, isJobFailed, isJobPending, isJobRunning } from 'hooks/api/util';

import { ImportActiveJob } from '../import-card-status/import-active-job/import-active-job.component';
import { ImportFailedJob } from '../import-card-status/import-failed-job/import-failed-job.component';
import { ImportJobDone } from '../import-card-status/import-job-done/import-job-done.component';

type LoadingImportDatasetProps = {
    jobId: string;
    size: number;
    fileName: string;
    stagedDatasetId: string;
    onSuccess: () => void;
    deleteEntry: () => void;
};

export const LoadingImportDataset = ({
    size,
    fileName,
    jobId,
    stagedDatasetId,
    onSuccess,
    deleteEntry,
}: LoadingImportDatasetProps) => {
    const { error, isError, data: job } = useImportJobStatus({ jobId, onSuccess });

    const isRunningOrPending = isJobRunning(job) || isJobPending(job);

    return (
        <View
            position='relative'
            borderColor='gray-200'
            borderRadius='regular'
            backgroundColor='gray-75'
            borderWidth='thin'
        >
            {isJobFailed(job) && (
                <ImportFailedJob
                    fileName={fileName}
                    size={size}
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
                    message={'An error occurred during import.'}
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

            {isJobDone(job) && (
                <ImportJobDone
                    size={size}
                    fileName={fileName}
                    stagedDatasetId={stagedDatasetId}
                    deleteEntry={deleteEntry}
                />
            )}
        </View>
    );
};
