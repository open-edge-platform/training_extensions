// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider, Flex, Loading, Text, View } from '@geti/ui';

import { ExportDatasetJob } from '../../../../../constants/shared-types';
import { BottomProgressBar } from '../../../../models/model-listing/current-model-training/bottom-progress-bar.component';
import { isJobRunning } from '../util';
import { CancelJobConfirmation } from './cancel-job-confirmation/cancel-job-confirmation.component';
import { ExportJobDetails } from './export-details/export-details.component';

type ExportActiveJobProps = {
    job: ExportDatasetJob;
};

export const ExportActiveJob = ({ job }: ExportActiveJobProps) => {
    const isRunning = isJobRunning(job);
    const progress = Math.max(0, Math.min(100, job?.progress ?? 0)) | 0;

    return (
        <BottomProgressBar progress={progress}>
            <View padding='size-150'>
                <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                    <ExportJobDetails metadata={job.metadata} />
                    <CancelJobConfirmation jobId={job.job_id} />
                </Flex>

                <Text>Dataset is being processed in order to export it</Text>

                <Divider size='S' marginY='size-150' />

                <Flex justifyContent='space-between'>
                    <Flex alignItems='center' gap='size-100'>
                        <Loading mode='inline' size='S' />
                        <Text>{job?.message}</Text>
                    </Flex>

                    {isRunning && <Text>{progress}%</Text>}
                </Flex>
            </View>
        </BottomProgressBar>
    );
};
