// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { View } from '@geti/ui';
import { isNil } from 'lodash-es';

import { useExportStatus } from '../hooks/use-export-status.hook';
import { isJobDone, isJobFailed, isJobPending, isJobRunning } from '../util';
import { ExportActiveJob } from './export-active-job.component';
import { ExportCompletedJob } from './export-completed-job/export-completed-job.component';
import { ExportFailedJob } from './export-failed-job/export-failed-job.component';

type ExportJobProps = {
    jobId: string;
};

export const ExportJob = ({ jobId }: ExportJobProps) => {
    const { data: job } = useExportStatus(jobId);

    const isRunningOrPending = isJobRunning(job) || isJobPending(job);

    if (isNil(job)) {
        return null;
    }

    return (
        <View
            position='relative'
            borderColor='gray-200'
            borderRadius='regular'
            backgroundColor='gray-75'
            borderWidth='thin'
        >
            {isJobDone(job) && <ExportCompletedJob job={job} />}
            {isJobFailed(job) && <ExportFailedJob job={job} />}
            {isRunningOrPending && <ExportActiveJob job={job} />}
        </View>
    );
};
