// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button } from '@geti/ui';
import { useDeleteStagedDataset } from 'hooks/api/staged-dataset.hook';

import { formatBytes } from '../../../shared/util';
import { JobStatusCard } from '../../job-status-card/job-status-card.component';

import classes from './import-failed-job.module.scss';

type ImportFailedJobProps = {
    size: number;
    error?: string;
    message?: string;
    fileName: string;
    stagedDatasetId: string;
    deleteEntry: () => void;
};

const TechnicalDetails = ({ error }: { error: string }) => (
    <details className={classes.details}>
        <summary className={classes.summary}>Technical details</summary>
        <pre className={classes.traceback}>{error}</pre>
    </details>
);

const BottomMessage = ({ error, message }: { error: string; message: string }) => {
    return (
        <>
            {message}
            <TechnicalDetails error={error} />
        </>
    );
};

export const ImportFailedJob = ({
    size,
    error,
    message,
    fileName,
    stagedDatasetId,
    deleteEntry,
}: ImportFailedJobProps) => {
    const deleteFileMutation = useDeleteStagedDataset({ stagedDatasetId, deleteEntry });

    const errorMessage = message ?? 'An unknown error occurred';

    return (
        <JobStatusCard
            title={`Import dataset - ${fileName} - ${formatBytes(size)}`}
            actionButtons={
                <Button
                    variant='secondary'
                    style='fill'
                    aria-label='close import dataset status'
                    onPress={deleteFileMutation.mutate}
                    isPending={deleteFileMutation.isPending}
                    isDisabled={deleteFileMutation.isPending}
                >
                    Close
                </Button>
            }
            bottomLeftMessage={error ? <BottomMessage error={error} message={errorMessage} /> : errorMessage}
        />
    );
};
