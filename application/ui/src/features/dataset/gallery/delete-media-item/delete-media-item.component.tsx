// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, DialogContainer, Tooltip, TooltipTrigger } from '@geti/ui';
import { Delete } from '@geti/ui/icons';

import { useDeleteMediaItem } from '../../api/use-delete-media-item';
import { AlertDialogContent } from './alert-dialog-content.component';

import classes from './delete-media-item.module.scss';

type DeleteMediaItemProps = {
    itemsIds: string[];
    onDeleted?: (deletedIds: string[]) => void;
};

export const DeleteMediaItem = ({ itemsIds = [], onDeleted }: DeleteMediaItemProps) => {
    const { deleteMedia, openDeleteDialog, closeDeleteDialog, isPending, isDeleteDialogOpen } = useDeleteMediaItem();

    const handleDelete = async () => {
        await deleteMedia(itemsIds, onDeleted);
    };

    return (
        <>
            <TooltipTrigger>
                <ActionButton
                    isQuiet
                    aria-label='delete media item'
                    isDisabled={isPending}
                    UNSAFE_className={classes.deleteButton}
                    onPress={openDeleteDialog}
                >
                    <Delete />
                </ActionButton>
                <Tooltip>Delete media item</Tooltip>
            </TooltipTrigger>

            <DialogContainer onDismiss={closeDeleteDialog}>
                {isDeleteDialogOpen && <AlertDialogContent itemsIds={itemsIds} onPrimaryAction={handleDelete} />}
            </DialogContainer>
        </>
    );
};
