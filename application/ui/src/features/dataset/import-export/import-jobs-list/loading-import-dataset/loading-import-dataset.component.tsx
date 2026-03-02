// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { View } from '@geti/ui';

import { useImportStatus } from '../../import-dataset/hooks/use-import-status.hook';
import { isJobDone, isJobFailed, isJobPending, isJobRunning } from '../../util';
import { ImportActiveJob } from '../import-active-job/import-active-job.component';
import { ImportFailedJob } from '../import-failed-job/import-failed-job.component';
import { ImportJobDone } from '../import-job-done/import-job-done.component';

type LoadingImportDatasetProps = {
    stagedDatasetId: string;
};

export const LoadingImportDataset = ({ stagedDatasetId }: LoadingImportDatasetProps) => {
    const { data: job, fileName, size } = useImportStatus({ stagedDatasetId });

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

            {isJobDone(job) && <ImportJobDone fileName={fileName} size={size} stagedDatasetId={stagedDatasetId} />}
        </View>
    );
};
