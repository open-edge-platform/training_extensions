// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, DialogContainer, toast } from '@geti/ui';
import { Delete } from '@geti/ui/icons';
import { useOverlayTriggerState } from '@react-stately/overlays';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isFunction } from 'lodash-es';

import { $api } from '../../../../api/client';
import { AlertDialogContent } from './alert-dialog-content.component';

import classes from './delete-media-item.module.scss';

type DeleteMediaItemProps = {
    itemsIds: string[];
    onDeleted?: (deletedIds: string[]) => void;
};

const isFulfilled = (response: PromiseSettledResult<{ itemId: string }>) => response.status === 'fulfilled';

export const DeleteMediaItem = ({ itemsIds = [], onDeleted }: DeleteMediaItemProps) => {
    const project_id = useProjectIdentifier();
    const alertDialogState = useOverlayTriggerState({});

    const removeMutation = $api.useMutation('delete', `/api/projects/{project_id}/dataset/items/{dataset_item_id}`, {
        meta: {
            invalidateQueries: [['get', '/api/projects/{project_id}/dataset/items']],
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

    const handleRemoveItems = async () => {
        alertDialogState.close();

        toast({ id: 'deleting-notification', type: 'info', message: `Deleting items...` });

        const deleteItemPromises = itemsIds.map(async (dataset_item_id) => {
            await removeMutation.mutateAsync({ params: { path: { project_id, dataset_item_id } } });

            return { itemId: dataset_item_id };
        });

        const responses = await Promise.allSettled(deleteItemPromises);
        const deletedIds = responses.filter(isFulfilled).map(({ value }) => value.itemId);

        isFunction(onDeleted) && onDeleted(deletedIds);

        toast({
            id: 'deleting-notification',
            type: 'success',
            message: `${deletedIds.length} item(s) deleted successfully`,
            duration: 3000,
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
