// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, View } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { usePrepareImportStatus } from '../import-dataset/hooks/use-prepare-import-status.hook';
import { isJobDone, isJobFailed, isJobPending, isJobRunning } from '../util';
import { ImportActiveJob } from './import-active-job/import-active-job.component';
import { ImportCompletedJob } from './import-completed-job/import-completed-job.component';
import { ImportFailedJob } from './import-failed-job/import-failed-job.component';

export const ImportJobsList = () => {
    const { data: job, fileName, size } = usePrepareImportStatus({});
    const isRunningOrPending = isJobRunning(job) || isJobPending(job);

    if (isEmpty(job)) {
        return null;
    }

    return (
        <Flex
            gap='size-250'
            direction='column'
            maxHeight='size-3400'
            marginBottom='size-250'
            UNSAFE_style={{ overflowY: 'auto' }}
        >
            <View
                position='relative'
                borderColor='gray-200'
                borderRadius='regular'
                backgroundColor='gray-75'
                borderWidth='thin'
            >
                {isJobDone(job) && <ImportCompletedJob job={job} fileName={String(fileName)} size={Number(size)} />}
                {isJobFailed(job) && <ImportFailedJob job={job} fileName={String(fileName)} size={Number(size)} />}
                {isRunningOrPending && <ImportActiveJob job={job} fileName={String(fileName)} size={Number(size)} />}
            </View>
        </Flex>
    );
};
