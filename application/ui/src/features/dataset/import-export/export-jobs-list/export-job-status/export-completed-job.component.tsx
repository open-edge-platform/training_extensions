// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Text, View } from '@geti/ui';
import { useLocalStorageDataset } from 'hooks/use-local-storage-dataset.hook';

import { ExportDatasetMetadata, Job } from '../../../../../constants/shared-types';
import { ExportDetails } from './export-details.component';

type ExportCompletedJobProps = {
    job: Job;
};

export const ExportCompletedJob = ({ job }: ExportCompletedJobProps) => {
    const { removeLsExportId } = useLocalStorageDataset();
    const metadata = job?.metadata as unknown as ExportDatasetMetadata;

    const handleClose = () => {
        removeLsExportId(job.job_id);
    };

    const handleDownload = () => {
        /* TODO: Implement download functionality https://github.com/open-edge-platform/training_extensions/pull/5443 */
        console.log('Download export job');
    };

    return (
        <View padding='size-150'>
            <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                <ExportDetails metadata={metadata} />

                <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                    <Button
                        variant='secondary'
                        style='fill'
                        aria-label='close export dataset status'
                        onPress={handleClose}
                    >
                        Close
                    </Button>
                    <Button variant='secondary' aria-label='download dataset' onPress={handleDownload}>
                        Download
                    </Button>
                </Flex>
            </Flex>

            <Text>Main Dataset is ready to download</Text>
        </View>
    );
};
