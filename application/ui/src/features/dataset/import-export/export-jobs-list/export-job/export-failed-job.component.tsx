// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Divider, Flex, Text, View } from '@geti/ui';

import { ExportDatasetJob } from '../../../../../constants/shared-types';
import { useExportDataset } from '../../../../../hooks/localStorage/use-export-dataset.hook';
import { ExportJobDetails } from './export-details.component';

type ExportFailedJobProps = {
    job: ExportDatasetJob;
};

export const ExportFailedJob = ({ job }: ExportFailedJobProps) => {
    const { removeLsExportId } = useExportDataset();

    const handleClose = () => {
        removeLsExportId(job.job_id);
    };
    console.log('failed job', job);

    return (
        <View padding='size-150'>
            <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                <ExportJobDetails metadata={job.metadata} />

                <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                    <Button
                        variant='secondary'
                        style='fill'
                        aria-label='close export dataset status'
                        onPress={handleClose}
                    >
                        Close
                    </Button>
                </Flex>
            </Flex>

            <Text>{job.message}</Text>
            <Divider size='S' marginY='size-150' />
            <Text>{job.error}</Text>
        </View>
    );
};
