// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider, Flex, Loading, Text, View } from '@geti/ui';
import { useExportDataset } from 'hooks/localStorage/use-export-dataset.hook';

import { ExportDatasetJob } from '../../../../../constants/shared-types';
import { BottomProgressBar } from '../../../../models/model-listing/current-model-training/bottom-progress-bar.component';
import { CancelJobConfirmation } from '../../cancel-job-confirmation/cancel-job-confirmation.component';
import { getJobProgress, isJobRunning } from '../../util';
import { ExportJobDetails } from './export-details/export-details.component';

type ExportActiveJobProps = {
    job: ExportDatasetJob;
    datasetName?: string;
};

export const ExportActiveJob = ({ job, datasetName }: ExportActiveJobProps) => {
    const isRunning = isJobRunning(job);
    const progress = getJobProgress(job?.progress);
    const { removeLsExportId } = useExportDataset();

    return (
        <BottomProgressBar progress={progress}>
            <View padding='size-150'>
                <Flex justifyContent='space-between' alignItems='center' gap='size-250'>
                    <ExportJobDetails metadata={job.metadata} datasetName={datasetName} />
                    <CancelJobConfirmation jobId={job.job_id} onRemove={async () => removeLsExportId(job.job_id)} />
                </Flex>

                <Text>Dataset is being processed in order to export it</Text>

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
