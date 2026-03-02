// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { View } from '@geti/ui';
import { useImportDatasetToProject } from 'hooks/localStorage/use-import-dataset-to-project.hook';

import { useImportJobStatus } from '../import-dataset/hooks/use-import-job-status.hook';
import { isJobFailed, isJobPending, isJobRunning } from '../util';
import { ImportActiveJob } from './import-active-job/import-active-job.component';
import { ImportFailedJob } from './import-failed-job/import-failed-job.component';

type PrepareImportDatasetProps = {
    stagedDatasetId: string;
};

export const PrepareImportDataset = ({ stagedDatasetId }: PrepareImportDatasetProps) => {
    const { getImportEntry, deleteImportEntry, updateImportEntryStep } = useImportDatasetToProject();
    const importLsEntry = getImportEntry(stagedDatasetId);

    const { data: job } = useImportJobStatus({
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
                <ImportFailedJob job={job} fileName={fileName} size={size} stagedDatasetId={stagedDatasetId} />
            )}

            {isRunningOrPending && (
                <ImportActiveJob job={job} fileName={fileName} size={size} stagedDatasetId={stagedDatasetId} />
            )}
        </View>
    );
};
