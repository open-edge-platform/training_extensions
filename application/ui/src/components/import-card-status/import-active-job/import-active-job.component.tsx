// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Divider, Flex, Loading, Text, View } from '@geti/ui';
import { useDeleteStagedDataset } from 'hooks/api/staged-dataset.hook';
import { getJobProgress, isJobRunning } from 'hooks/api/util';

import { Job } from '../../../constants/shared-types';
import { CancelJobConfirmation } from '../../../features/dataset/import-export/cancel-job-confirmation/cancel-job-confirmation.component';
import { BottomProgressBar } from '../../../features/models/model-listing/current-model-training/bottom-progress-bar.component';
import { formatBytes } from '../../../shared/util';

type ImportActiveJobProps = {
    job: Job;
    size: number;
    fileName: string;
    stagedDatasetId: string;
    deleteEntry: () => void;
};

export const ImportActiveJob = ({ job, size, fileName, stagedDatasetId, deleteEntry }: ImportActiveJobProps) => {
    const deleteFileMutation = useDeleteStagedDataset({ stagedDatasetId, deleteEntry });

    const isRunning = isJobRunning(job);
    const progress = getJobProgress(job?.progress);

    const handleRemove = () => {
        return deleteFileMutation.mutateAsync();
    };

    return (
        <BottomProgressBar progress={progress}>
            <View padding='size-150'>
                <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                    <Text UNSAFE_style={{ fontWeight: 500, fontSize: dimensionValue('size-200') }}>
                        Import dataset - {fileName} - {formatBytes(size)}
                    </Text>

                    <CancelJobConfirmation jobId={job.job_id} onRemove={handleRemove} />
                </Flex>

                <Text>{fileName} file is being processed for import</Text>

                <Divider size='S' marginY='size-150' />

                <Flex justifyContent='space-between'>
                    <Flex alignItems='center' gap='size-100'>
                        <Loading mode='inline' size='S' />
                        <Text>{job?.message ?? job.status.toLocaleLowerCase()}</Text>
                    </Flex>

                    {isRunning && <Text>{progress}%</Text>}
                </Flex>
            </View>
        </BottomProgressBar>
    );
};
