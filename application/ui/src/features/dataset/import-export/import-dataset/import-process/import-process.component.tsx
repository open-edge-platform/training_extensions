// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Flex, Loading, Text } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { CircularProgress } from '../../../../../components/circular-progress/circular-progress.component';
import { useImportDatasetToProject } from '../../../../../hooks/localStorage/use-import-dataset-to-project.hook';
import { useImportDatasetDialogState } from '../../../providers/export-import-dataset-dialog-provider.component';
import { getJobProgress, isJobPending } from '../../util';
import { useImportJobStatus } from '../hooks/use-import-job-status.hook';

import classes from './import-process.module.scss';

type ImportProcessProps = {
    currentStagedId: string;
};

export const ImportProcess = ({ currentStagedId }: ImportProcessProps) => {
    const { setCurrentStep } = useImportDatasetDialogState();
    const { getImportEntry, updateImportEntryStep, deleteImportEntry } = useImportDatasetToProject();
    const importLsEntry = getImportEntry(currentStagedId);

    const {
        data: job,
        isFetching,
        isPending,
    } = useImportJobStatus({
        jobId: importLsEntry?.prepareJobId,
        onError: () => {
            setCurrentStep('uploading');
            deleteImportEntry(currentStagedId);
        },
        onSuccess: () => {
            setCurrentStep('labelMapping');
            updateImportEntryStep(currentStagedId, 'labelMapping');
        },
    });

    const fileName = importLsEntry?.fileName ?? '';
    const progress = getJobProgress(job?.progress);
    const isPreparingJobLoading = isJobPending(job) && isPending;

    if (!isFetching && isEmpty(job)) {
        return null;
    }

    return (
        <Flex
            width='100%'
            height='100%'
            gap='size-275'
            direction='column'
            alignItems='center'
            justifyContent='center'
            UNSAFE_style={{ padding: dimensionValue('size-500') }}
        >
            {!isPreparingJobLoading && (
                <CircularProgress
                    size={80}
                    percentage={progress}
                    strokeWidth={8}
                    labelFontSize={12}
                    color='static-blue-200'
                    labelFontColor='gray-700'
                    backStrokeColor='gray-75'
                />
            )}

            {isPreparingJobLoading && <Loading mode='inline' size='L' style={{ height: 'auto' }} />}

            <Flex direction='column' alignItems='center' justifyContent='center'>
                <Text UNSAFE_className={classes.title}>Preparing</Text>
                <Text UNSAFE_className={classes.description}>Prepare dataset import to existing project</Text>

                <Text marginTop='size-100'>{fileName}</Text>
            </Flex>
        </Flex>
    );
};
