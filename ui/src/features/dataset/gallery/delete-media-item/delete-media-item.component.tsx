// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef } from 'react';

import { ActionButton, DialogContainer, removeToast, toast } from '@geti/ui';
import { Delete } from '@geti/ui/icons';
import { useOverlayTriggerState } from '@react-stately/overlays';

import { $api } from '../../../../api/client';
import { useProjectIdentifier } from '../../../../hooks/use-project-identifier.hook';
import { AlertDialogContent } from './alert-dialog-content.component';

import classes from './delete-media-item.module.scss';

type DeleteMediaItemProps = {
    itemId: string;
};

export const DeleteMediaItem = ({ itemId }: DeleteMediaItemProps) => {
    const project_id = useProjectIdentifier();
    const processingToastId = useRef<null | string | number>(null);
    const alertDialogState = useOverlayTriggerState({});

    const removeMutation = $api.useMutation('delete', `/api/projects/{project_id}/dataset/items/{dataset_item_id}`, {
        onSuccess: () => {
            toast({ type: 'success', message: `Item "${itemId}" was deleted successfully`, duration: 3000 });
        },
        onError: (error) => {
            toast({ type: 'error', message: `Failed to delete item: ${error?.detail}` });
        },
        onSettled: () => {
            if (processingToastId.current) {
                removeToast(processingToastId.current);
            }
        },
    });

    const handleRemoveItems = () => {
        alertDialogState.close();

        removeMutation.mutate({ params: { path: { project_id, dataset_item_id: itemId } } });

        processingToastId.current = toast({
            type: 'info',
            message: `Deleting ${itemId} item(s)...`,
        });
    };

    return (
        <>
            <ActionButton
                isQuiet
                aria-label='delete media item'
                isDisabled={removeMutation.isPending}
                UNSAFE_className={classes.deleteButton}
                onPress={alertDialogState.open}
            >
                <Delete />
            </ActionButton>

            <DialogContainer onDismiss={alertDialogState.close}>
                {alertDialogState.isOpen && <AlertDialogContent itemId={itemId} onPrimaryAction={handleRemoveItems} />}
            </DialogContainer>
        </>
    );
};
