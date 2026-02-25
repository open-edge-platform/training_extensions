// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Divider, Flex, Loading, Text, View } from '@geti/ui';
import { usePrepareImportDataset } from 'hooks/localStorage/use-prepare-import-dataset.hook';

import { PrepareImportDatasetJob } from '../../../../../constants/shared-types';
import { formatBytes } from '../../../../../shared/util';
import { BottomProgressBar } from '../../../../models/model-listing/current-model-training/bottom-progress-bar.component';
import { CancelJobConfirmation } from '../../cancel-job-confirmation/cancel-job-confirmation.component';
import { getJobProgress, isJobRunning } from '../../util';

type ImportActiveJobProps = {
    size: number;
    fileName: string;
    job: PrepareImportDatasetJob;
};

export const ImportActiveJob = ({ job, fileName, size }: ImportActiveJobProps) => {
    const isRunning = isJobRunning(job);
    const progress = getJobProgress(job?.progress);
    const { removeLsPreparingImport } = usePrepareImportDataset();

    return (
        <BottomProgressBar progress={progress}>
            <View padding='size-150'>
                <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                    <Text UNSAFE_style={{ fontWeight: 500, fontSize: dimensionValue('size-200') }}>
                        Import dataset - {fileName} - {formatBytes(size)}
                    </Text>

                    <CancelJobConfirmation jobId={job.job_id} onRemove={removeLsPreparingImport} />
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
