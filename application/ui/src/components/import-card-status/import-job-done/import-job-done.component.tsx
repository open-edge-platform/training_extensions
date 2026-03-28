// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button } from '@geti/ui';
import { CheckCircleOutlined } from '@geti/ui/icons';
import { useDeleteStagedDataset } from 'hooks/api/staged-dataset.hook';

import { formatBytes } from '../../../shared/util';
import { JobStatusCard } from '../../job-status-card/job-status-card.component';

import classes from './import-job-done.module.scss';

type ImportJobDoneProps = {
    size: number;
    fileName: string;
    stagedDatasetId: string;
    deleteEntry: () => void;
};

export const ImportJobDone = ({ fileName, size, stagedDatasetId, deleteEntry }: ImportJobDoneProps) => {
    const deleteFileMutation = useDeleteStagedDataset({ stagedDatasetId, deleteEntry });

    const handleClose = () => {
        deleteFileMutation.mutate();
    };

    return (
        <JobStatusCard
            title={`Import dataset - ${fileName} - ${formatBytes(size)}`}
            actionButtons={
                <Button
                    variant='secondary'
                    style='fill'
                    aria-label='close import dataset status'
                    onPress={handleClose}
                    isPending={deleteFileMutation.isPending}
                    isDisabled={deleteFileMutation.isPending}
                >
                    Close
                </Button>
            }
            message={`${fileName} file has been imported successfully`}
            bottomLeftMessage={'Ready'}
            bottomIcon={<CheckCircleOutlined className={classes.checkIcon} width={16} height={16} />}
        />
    );
};
