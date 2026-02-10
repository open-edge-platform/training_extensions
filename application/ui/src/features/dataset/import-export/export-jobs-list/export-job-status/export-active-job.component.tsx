// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Divider, Flex, Loading, Text, View } from '@geti/ui';

import { ThinProgressBar } from '../../../../../components/thin-progress-bar/thin-progress-bar.component';
import { ExportDatasetMetadata, Job } from '../../../../../constants/shared-types';
import { isJobRunning } from '../util';
import { ExportDetails } from './export-details.component';

type ExportActiveJobProps = {
    job: Job;
};

export const ExportActiveJob = ({ job }: ExportActiveJobProps) => {
    const progress = Math.max(0, Math.min(100, job?.progress ?? 0)) | 0;
    const metadata = job?.metadata as unknown as ExportDatasetMetadata;

    const handleCancel = () => {
        console.log('Cancel export job');
    };

    return (
        <>
            <View padding='size-150'>
                <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                    <ExportDetails metadata={metadata} />

                    <Button variant='negative' style='outline' onPress={handleCancel}>
                        Cancel
                    </Button>
                </Flex>

                <Text>Main dataset is being processed in order to export it</Text>

                <Divider size='S' marginY='size-150' />

                <Flex justifyContent='space-between'>
                    <Flex alignItems='center' gap='size-100'>
                        {isJobRunning(job) && <Loading mode='inline' size='S' />}
                        <Text>{job?.message}</Text>
                    </Flex>

                    {isJobRunning(job) && <Text>{progress}%</Text>}
                </Flex>
            </View>

            {isJobRunning(job) && (
                <View position='absolute' left={0} right={0} bottom={0}>
                    <ThinProgressBar size='size-25' customColor='var(--energy-blue-shade)' progress={progress} />
                </View>
            )}
        </>
    );
};
