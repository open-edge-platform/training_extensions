// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { View } from '@geti/ui';

import { usePrepareImportStatus } from '../import-dataset/hooks/use-prepare-import-status.hook';
import { isJobFailed, isJobPending, isJobRunning } from '../util';
import { ImportActiveJob } from './import-active-job/import-active-job.component';
import { ImportFailedJob } from './import-failed-job/import-failed-job.component';

type PrepareImportDatasetProps = {
    stagedDatasetId: string;
};

export const PrepareImportDataset = ({ stagedDatasetId }: PrepareImportDatasetProps) => {
    const { data: job, fileName, size } = usePrepareImportStatus({ stagedDatasetId, onSuccess: () => {} });

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
