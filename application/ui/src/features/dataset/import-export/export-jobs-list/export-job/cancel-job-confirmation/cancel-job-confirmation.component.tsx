// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { AlertDialog, Button, DialogTrigger } from '@geti/ui';
import { useOverlayTriggerState } from '@react-stately/overlays';
import { useLocalStorageDataset } from 'hooks/use-local-storage-dataset.hook';

import { $api } from '../../../../../../api/client';
import { isInvalidJob } from '../../util';

type CancelJobConfirmationProps = {
    jobId: string;
};

export const CancelJobConfirmation = ({ jobId }: CancelJobConfirmationProps) => {
    const dialogState = useOverlayTriggerState({});
    const { removeLsExportId } = useLocalStorageDataset();
    const cancelMutation = $api.useMutation('post', `/api/jobs/{job_id}:cancel`);

    const handleCancel = () => {
        cancelMutation.mutate(
            { params: { path: { job_id: jobId } } },
            {
                onSuccess: () => {
                    removeLsExportId(jobId);
                },
                onError: (error) => {
                    isInvalidJob(error) && removeLsExportId(jobId);
                },
                onSettled: () => {
                    dialogState.close();
                },
            }
        );
    };

    return (
        <DialogTrigger>
            <Button variant='negative' style='outline' aria-label='cancel job dialog'>
                Cancel
            </Button>
            <AlertDialog
                title='Cancel Job'
                variant='destructive'
                cancelLabel='Cancel'
                autoFocusButton='primary'
                primaryActionLabel='Cancel Job'
                onPrimaryAction={handleCancel}
                onSecondaryAction={dialogState.close}
                isPrimaryActionDisabled={cancelMutation.isPending}
            >
                {`Are you sure you want to cancel the job "${jobId}"?`}
            </AlertDialog>
        </DialogTrigger>
    );
};
