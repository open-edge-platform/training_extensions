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

const useExportStatus = (job_id: string) => {
    return {
        job: getMockedJob({
            job_id,
            progress: 89.99999999,
            message: 'Exporting dataset',
            status: 'PENDING',
            metadata: {
                project_id: '456',
                export_format: 'coco',
                staged_dataset_id: '123',
                filters: {
                    include_unannotated: true,
                    labels: [
                        '359df657-9e93-4dca-a3d6-52fce0f1b3a9',
                        '05b6dd25-4451-4382-8fb2-fd65efb76c2a',
                        '284aede6-9ee8-4b38-b602-8f647b7521b1',
                        '1f144258-622b-4a72-93e0-f32ef3d30266',
                        '2b20b64b-eccb-4c6c-8b10-a6f7a9d29d5a',
                        'fcff3437-3436-4443-9ee5-087ca34924b7',
                        'b297fb7c-36f7-499e-98ec-0088a3ffd644',
                        '65a59fd3-eb40-4192-b5fa-af43d24c437c',
                        'ff0d6fa0-23f8-4ac4-8610-6fa20db5a9bc',
                        '47cc6f96-dfb4-4bc8-ac81-9b5cca10c6e7',
                        'b9548539-f5d1-4525-aaea-758c8bfe1b95',
                        '92c6c686-be2b-4210-93aa-a4876c416fe5',
                        '10e00881-91b2-42e1-bd9e-004a09a79197',
                        '7eacb6cf-2618-40d1-bd7b-1f5458a4ed55',
                        '37020f2f-dc29-4cff-981e-81b1eba0a78c',
                        '998c3e6f-d324-4176-a90f-1e8cf0a2f25d',
                        'cc1be5a4-612f-4e70-9d21-c4efa84f9e9a',
                        '1fdfb457-adb5-45fc-a195-f3ecd366ddb1',
                        '444fbdf5-23f0-4e21-bb06-f6873aca0768',
                        'aac2432d-3cfc-4366-a9c4-5031f5158599',
                    ],
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
