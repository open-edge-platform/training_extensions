// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { AlertDialog, Button, DialogTrigger } from '@geti/ui';
import { useOverlayTriggerState } from '@react-stately/overlays';
import { isInvalidJob } from 'hooks/api/util';

import { $api } from '../../../../api/client';

type CancelJobConfirmationProps = {
    jobId: string;
    onRemove: () => void | Promise<void>;
};

export const CancelJobConfirmation = ({ jobId, onRemove }: CancelJobConfirmationProps) => {
    const dialogState = useOverlayTriggerState({});
    const cancelMutation = $api.useMutation('post', `/api/jobs/{job_id}:cancel`);

    const handleCancel = () => {
        cancelMutation.mutate(
            { params: { path: { job_id: jobId } } },
            {
                onSuccess: async () => await onRemove(),
                onError: async (error) => {
                    isInvalidJob(error) && (await onRemove());
                },
                onSettled: () => {
                    dialogState.close();
                },
            }
        );
    };

    return (
        <DialogTrigger>
            <Button
                variant='negative'
                style='outline'
                aria-label='cancel job dialog'
                isDisabled={cancelMutation.isPending}
                isPending={cancelMutation.isPending}
            >
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
