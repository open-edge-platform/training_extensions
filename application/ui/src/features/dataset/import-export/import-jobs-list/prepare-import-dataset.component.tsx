// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { View } from '@geti/ui';
import { useImportJobStatus } from 'hooks/api/jobs/use-import-job-status.hook';
import { isJobFailed, isJobPending, isJobRunning } from 'hooks/api/util';
import { useImportDatasetToProject } from 'hooks/localStorage/use-import-dataset-to-project.hook';

import { ImportActiveJob } from './import-active-job/import-active-job.component';
import { ImportFailedJob } from './import-failed-job/import-failed-job.component';

type PrepareImportDatasetProps = {
    stagedDatasetId: string;
};

export const PrepareImportDataset = ({ stagedDatasetId }: PrepareImportDatasetProps) => {
    const { getImportEntry, deleteImportEntry, updateImportEntryStep } = useImportDatasetToProject();
    const importLsEntry = getImportEntry(stagedDatasetId);

    const {
        data: job,
        isError,
        error,
    } = useImportJobStatus({
        jobId: importLsEntry?.prepareJobId,
        onError: () => {
            deleteImportEntry(stagedDatasetId);
        },
        onSuccess: () => {
            updateImportEntryStep(stagedDatasetId, 'labelMapping');
        },
    });

    const size = importLsEntry?.size ?? 0;
    const fileName = importLsEntry?.fileName ?? '';
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
                    size={size}
                    fileName={fileName}
                    error={job.error ?? ''}
                    message={job.message ?? ''}
                    stagedDatasetId={stagedDatasetId}
                />
            )}

            {isError && (
                <ImportFailedJob
                    size={size}
                    fileName={fileName}
                    error={`${error?.detail ?? 'Unknown error'}`}
                    message={'An error occurred during import preparation.'}
                    stagedDatasetId={stagedDatasetId}
                />
            )}

            {isRunningOrPending && (
                <ImportActiveJob job={job} fileName={fileName} size={size} stagedDatasetId={stagedDatasetId} />
            )}
        </View>
    );
};
