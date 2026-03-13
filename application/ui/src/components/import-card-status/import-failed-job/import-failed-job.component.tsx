// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button } from '@geti/ui';
import { useDeleteStagedDataset } from 'hooks/api/staged-dataset.hook';

import { formatBytes } from '../../../shared/util';
import { JobStatusCard } from '../../job-status-card/job-status-card.component';

type ImportFailedJobProps = {
    size: number;
    error?: string;
    message?: string;
    fileName: string;
    stagedDatasetId: string;
    deleteEntry: () => void;
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

    const handleClose = () => {
        deleteFileMutation.mutate();
    };

    return (
        <JobStatusCard
            title={`Import dataset - ${fileName} - ${formatBytes(size)}`}
            actionButtons={
                <Button variant='secondary' style='fill' aria-label='close import dataset status' onPress={handleClose}>
                    Close
                </Button>
            }
            message={message}
            bottomLeftMessage={error ?? 'An unknown error '}
        />
    );
};
