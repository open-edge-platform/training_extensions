// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@geti/ui';
import { useImportJobStatus } from 'hooks/api/jobs/use-import-job-status.hook';
import { useDeleteStagedDataset } from 'hooks/api/staged-dataset.hook';
import { isJobFailed, isJobPending, isJobRunning } from 'hooks/api/util';

import { formatBytes } from '../../shared/util';
import { ImportActiveJob } from '../import-card-status/import-active-job/import-active-job.component';
import { ImportFailedJob } from '../import-card-status/import-failed-job/import-failed-job.component';

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
    const deleteFileMutation = useDeleteStagedDataset({ stagedDatasetId });

    const {
        error,
        isError,
        data: job,
    } = useImportJobStatus({
        jobId,
        onSuccess: () => {
            deleteEntry();
            deleteFileMutation.mutate();
            onSuccess();

            toast({
                message: `Dataset ${fileName} ${formatBytes(size)} imported successfully.`,
                type: 'success',
            });
        },
    });

    const isRunningOrPending = isJobRunning(job) || isJobPending(job);

    return (
        <>
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
        </>
    );
};
