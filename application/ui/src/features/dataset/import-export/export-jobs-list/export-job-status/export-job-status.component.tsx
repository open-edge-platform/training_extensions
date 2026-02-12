// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { View } from '@geti/ui';
import { isNil } from 'lodash-es';
import { getMockedJob } from 'mocks/mock-job';

import { isJobDone, isJobPending, isJobRunning } from '../util';
import { ExportActiveJob } from './export-active-job.component';
import { ExportCompletedJob } from './export-completed-job.component';

type ExportJobStatusProps = {
    jobId: string;
};

/* TODO: Update once https://github.com/open-edge-platform/training_extensions/pull/5443 gets merged*/
const useExportStatus = (job_id: string) => {
    return {
        job: getMockedJob({
            job_id,
            progress: 89.99999999,
            message: 'Exporting dataset',
            status: 'PENDING',
            metadata: {
                project_id: '456',
                staged_dataset_id: '123',
                filters: {
                    include_unannotated: true,
                    labels: [],
                },
            },
        }),
        stagedDatasetId: '123',
        isFetching: false,
    };
};

export const ExportJobStatus = ({ jobId }: ExportJobStatusProps) => {
    const { job } = useExportStatus(jobId);

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
            {isRunningOrPending && <ExportActiveJob job={job} />}
        </View>
    );
};
