// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Text, View } from '@geti/ui';

import { ExportDatasetJob } from '../../../../../constants/shared-types';
import { useExportDataset } from '../../../../../hooks/localStorage/use-export-dataset.hook';
import { ExportJobDetails } from './export-details.component';

type ExportCompletedJobProps = {
    job: ExportDatasetJob;
};

export const ExportCompletedJob = ({ job }: ExportCompletedJobProps) => {
    const { removeLsExportId } = useExportDataset();

    const handleClose = () => {
        removeLsExportId(job.job_id);
    };

    const handleDownload = () => {
        /* TODO: Implement download functionality https://github.com/open-edge-platform/training_extensions/pull/5443 */
    };

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
                    <Button variant='secondary' aria-label='download dataset' onPress={handleDownload}>
                        Download
                    </Button>
                </Flex>
            </Flex>

            <Text>Main Dataset is ready to download</Text>
        </View>
    );
};
