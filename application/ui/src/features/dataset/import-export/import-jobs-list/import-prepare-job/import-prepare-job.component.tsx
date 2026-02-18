// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider, Flex, Loading, Text, View } from '@geti/ui';

import { PrepareImportDatasetJob } from '../../../../../constants/shared-types';
import { BottomProgressBar } from '../../../../models/model-listing/current-model-training/bottom-progress-bar.component';
import { CancelJobConfirmation } from '../../export-jobs-list/export-job/cancel-job-confirmation/cancel-job-confirmation.component';

type ImportPrepareJobProps = {
    job: PrepareImportDatasetJob;
};
export const ImportPrepareJob = ({ job }: ImportPrepareJobProps) => {
    const progress = 50;

    return (
        <View
            position='relative'
            borderColor='gray-200'
            borderRadius='regular'
            backgroundColor='gray-75'
            borderWidth='thin'
        >
            <BottomProgressBar progress={progress}>
                <View padding='size-150'>
                    <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                        details
                        <CancelJobConfirmation jobId={job.job_id} />
                    </Flex>

                    <Text>Main Dataset is being processed in order to export it</Text>

                    <Divider size='S' marginY='size-150' />

                    <Flex justifyContent='space-between'>
                        <Flex alignItems='center' gap='size-100'>
                            {<Loading mode='inline' size='S' />}
                            {/* <Text>{job?.message}</Text> */}
                        </Flex>

                        {<Text>{progress}%</Text>}
                    </Flex>
                </View>
            </BottomProgressBar>
        </View>
    );
};
