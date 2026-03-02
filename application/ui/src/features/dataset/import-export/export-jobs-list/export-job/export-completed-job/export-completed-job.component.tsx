// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Text, View } from '@geti/ui';
import { isNil } from 'lodash-es';

import { $api, API_BASE_URL } from '../../../../../../api/client';
import { ExportDatasetJob } from '../../../../../../constants/shared-types';
import { useExportDataset } from '../../../../../../hooks/localStorage/use-export-dataset.hook';
import { downloadFile } from '../../../../../../shared/util';
import { isInvalidStagedFile } from '../../../util';
import { ExportJobDetails } from '../export-details/export-details.component';

type ExportCompletedJobProps = {
    job: ExportDatasetJob;
    datasetName?: string;
};

export const ExportCompletedJob = ({ job, datasetName }: ExportCompletedJobProps) => {
    const { removeLsExportId } = useExportDataset();
    const stageDatasetResponse = $api.useQuery('get', '/api/staged_datasets/{staged_dataset_id}', {
        params: { path: { staged_dataset_id: job.metadata.dataset_id } },
    });

    const removeStagedDatasetMutation = $api.useMutation('delete', '/api/staged_datasets/{staged_dataset_id}');

    const handleClose = () => {
        removeStagedDatasetMutation.mutate(
            { params: { path: { staged_dataset_id: job.metadata.dataset_id } } },
            {
                onSuccess: () => removeLsExportId(job.job_id),
                onError: (error) => {
                    isInvalidStagedFile(error) && removeLsExportId(job.job_id);
                },
            }
        );
    };

    const handleDownload = () => {
        const url = `${API_BASE_URL}/api/staged_datasets/${job.metadata.dataset_id}/zip`;

        downloadFile(url, `dataset_${job.metadata.dataset_id}.zip`);
    };

    return (
        <View padding='size-150'>
            <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                <ExportJobDetails metadata={job.metadata} datasetName={datasetName} />

                <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                    <Button
                        variant='secondary'
                        style='fill'
                        aria-label='close export dataset status'
                        onPress={handleClose}
                        isPending={removeStagedDatasetMutation.isPending}
                        isDisabled={removeStagedDatasetMutation.isPending}
                    >
                        Close
                    </Button>
                    <Button
                        variant='secondary'
                        aria-label='download dataset'
                        onPress={handleDownload}
                        isPending={stageDatasetResponse.isFetching}
                        isDisabled={
                            stageDatasetResponse.isFetching ||
                            removeStagedDatasetMutation.isPending ||
                            isNil(job.metadata.dataset_id)
                        }
                    >
                        Download
                    </Button>
                </Flex>
            </Flex>

            <Text>Dataset is ready to download</Text>
        </View>
    );
};
