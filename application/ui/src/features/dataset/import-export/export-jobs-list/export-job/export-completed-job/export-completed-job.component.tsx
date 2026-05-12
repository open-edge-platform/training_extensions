// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Divider, Flex, Text, toast, View } from '@geti/ui';
import { useDeleteStagedDataset, useStagedDataset } from 'hooks/api/staged-dataset.hook';
import { isJobDone } from 'hooks/api/util';
import { isNil } from 'lodash-es';

import { API_BASE_URL } from '../../../../../../api/client';
import { ExportDatasetJob } from '../../../../../../constants/shared-types';
import { useExportDataset } from '../../../../../../hooks/storage/use-export-dataset.hook';
import { downloadFile, isNonEmptyString } from '../../../../../../shared/util';
import { ExportJobDetails } from '../export-details/export-details.component';

type ExportCompletedJobProps = {
    job: ExportDatasetJob;
    datasetName?: string;
};

export const ExportCompletedJob = ({ job, datasetName }: ExportCompletedJobProps) => {
    const { removeLsExportId } = useExportDataset();
    const stageDatasetResponse = useStagedDataset(job.metadata.dataset_id);

    const hasInvalidStagedDataset = isNil(job.metadata.dataset_id);
    const isDoneWithEmptyStagedFile = isJobDone(job) && hasInvalidStagedDataset;
    const message = isDoneWithEmptyStagedFile ? job.message : 'Dataset is ready for download';

    const removeStagedDatasetMutation = useDeleteStagedDataset({
        stagedDatasetId: job.metadata.dataset_id,
        deleteEntry: () => removeLsExportId(job.job_id),
    });

    const handleClose = () => {
        if (isNonEmptyString(job.metadata.dataset_id)) {
            removeStagedDatasetMutation.mutate();
        } else {
            removeLsExportId(job.job_id);
        }
    };

    const handleDownload = () => {
        const url = `${API_BASE_URL}/api/staged_datasets/${job.metadata.dataset_id}/zip`;

        downloadFile(url, `dataset_${job.metadata.dataset_id}.zip`);
        toast({ type: 'info', message: 'Dataset download started' });
    };

    return (
        <View padding='size-150'>
            <Flex justifyContent='space-between' alignItems='end' gap='size-250' marginBottom='size-125'>
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
                            hasInvalidStagedDataset ||
                            stageDatasetResponse.isFetching ||
                            removeStagedDatasetMutation.isPending
                        }
                    >
                        Download
                    </Button>
                </Flex>
            </Flex>

            <Divider size={'S'} marginY={'size-150'} />

            <Text>{message}</Text>
        </View>
    );
};
