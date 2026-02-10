// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Text, View } from '@geti/ui';

import { ExportDatasetMetadata, Job } from '../../../../../constants/shared-types';
import { ExportDetails } from './export-details.component';

type ExportCompletedJobProps = {
    job: Job;
};

export const ExportCompletedJob = ({ job }: ExportCompletedJobProps) => {
    const metadata = job?.metadata as unknown as ExportDatasetMetadata;

    const handleClose = () => {
        console.log('Close export job');
    };

    const handleDownload = () => {
        console.log('Download export job');
    };

    return (
        <View padding='size-150'>
            <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                <ExportDetails metadata={metadata} />

                <Button variant='secondary' style='fill' aria-label='close export dataset status' onPress={handleClose}>
                    Close
                </Button>
                <Button variant='secondary' aria-label='download dataset' width='size-3000' onPress={handleDownload}>
                    Download
                </Button>
            </Flex>

            <Text>Main Dataset is ready to download</Text>
        </View>
    );
};
