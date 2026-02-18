// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Flex, Loading, Text } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { CircularProgress } from '../../../../../components/circular-progress/circular-progress.component';
import { isJobPending } from '../../export-jobs-list/util';
import { usePrepareImportStatus } from '../hooks/use-prepare-import-status.hook';
import { ImportDatasetState } from '../util';

import classes from './import-process.module.scss';

type ImportProcessProps = {
    onNextStep: (step: ImportDatasetState) => void;
};

export const ImportProcess = ({ onNextStep }: ImportProcessProps) => {
    const {
        data: job,
        isFetching,
        fileName,
    } = usePrepareImportStatus({
        onError: () => onNextStep('dropzone'),
    });

    const progress = Math.max(0, Math.min(100, job?.progress ?? 0)) | 0;
    const isPreparingJobLoading = isJobPending(job) && isFetching;
    console.log('--> job', job);
    console.log('progress', progress);

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
