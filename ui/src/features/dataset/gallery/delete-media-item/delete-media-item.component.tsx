// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, DialogContainer, toast } from '@geti/ui';
import { Delete } from '@geti/ui/icons';
import { useOverlayTriggerState } from '@react-stately/overlays';
import { isFunction } from 'lodash-es';

import { $api } from '../../../../api/client';
import { useProjectIdentifier } from '../../../../hooks/use-project-identifier.hook';
import { AlertDialogContent } from './alert-dialog-content.component';

import classes from './delete-media-item.module.scss';

type DeleteMediaItemProps = {
    itemsIds: string[];
    onDeleted?: (deletedIds: string[]) => void;
};

export const DeleteMediaItem = ({ itemsIds = [], onDeleted }: DeleteMediaItemProps) => {
    const project_id = useProjectIdentifier();
    const alertDialogState = useOverlayTriggerState({});

    const removeMutation = $api.useMutation('delete', `/api/projects/{project_id}/dataset/items/{dataset_item_id}`, {
        onSuccess: (_, { params: { path } }) => {
            const { dataset_item_id: itemId } = path;

            toast({
                id: itemId,
                type: 'success',
                message: `Item "${itemId}" was deleted successfully`,
                duration: 3000,
            });
        },
        onError: (error, { params: { path } }) => {
            const { dataset_item_id: itemId } = path;

            toast({
                id: itemId,
                type: 'error',
                message: `Failed to delete, ${error?.detail}`,
            });
        },
    });

    const handleRemoveItems = () => {
        alertDialogState.close();

        const deleteItemPromises = itemsIds.map(async (itemId) => {
            toast({ id: itemId, type: 'info', message: `Deleting item "${itemId}"...` });

            await removeMutation.mutateAsync({
                params: { path: { project_id, dataset_item_id: itemId } },
            });

            return { itemId };
        });

        Promise.allSettled(deleteItemPromises).then((responses) => {
            const deletedIds = responses.filter((res) => res.status === 'fulfilled').map(({ value }) => value.itemId);
            isFunction(onDeleted) && onDeleted(deletedIds);
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
                {alertDialogState.isOpen && (
                    <AlertDialogContent itemsIds={itemsIds} onPrimaryAction={handleRemoveItems} />
                )}
            </DialogContainer>
        </>
    );
};
