// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Text, View } from '@geti/ui';

import { $api } from '../../../../../api/client';
import { ExportDatasetJob } from '../../../../../constants/shared-types';
import { useExportDataset } from '../../../../../hooks/localStorage/use-export-dataset.hook';
import { downloadFile, formatDownloadUrl } from '../../../../../shared/util';
import { ExportJobDetails } from './export-details.component';

type ExportCompletedJobProps = {
    job: ExportDatasetJob;
};

export const ExportCompletedJob = ({ job }: ExportCompletedJobProps) => {
    const { removeLsExportId } = useExportDataset();
    const stageDatasetResponse = $api.useQuery('get', '/api/staged_datasets/{staged_dataset_id}', {
        params: { path: { staged_dataset_id: job.metadata.dataset_id } },
    });
    /*     const stagedFileResponse = $api.useQuery(
        'get',
        '/api/staged_datasets/{staged_dataset_id}/zip',
        {
            params: { path: { staged_dataset_id: job.metadata.dataset_id } },
        },
        {
            enabled: stageDatasetResponse?.data?.ready_for_export,
        }
    ); */

    console.log('status', stageDatasetResponse.data);
    const handleClose = () => {
        removeLsExportId(job.job_id);
    };

    const handleDownload = () => {
        downloadFile(
            formatDownloadUrl(`/api/staged_datasets/${job.metadata.dataset_id}/zip`),
            `dataset_${job.metadata.dataset_id}.zip`
        );
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
                    <Button
                        variant='secondary'
                        aria-label='download dataset'
                        onPress={handleDownload}
                        isPending={stageDatasetResponse.isFetching}
                        isDisabled={stageDatasetResponse.isFetching}
                    >
                        Download
                    </Button>
                </Flex>
            </Flex>

            <Text>Dataset is ready to download</Text>
        </View>
    );
};
