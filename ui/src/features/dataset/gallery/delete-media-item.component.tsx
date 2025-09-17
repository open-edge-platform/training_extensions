// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef } from 'react';

import { ActionButton, AlertDialog, DialogTrigger, removeToast, toast } from '@geti/ui';
import { Delete } from '@geti/ui/icons';

import { $api } from '../../../api/client';
import { useProjectIdentifier } from '../../../hooks/use-project-identifier.hook';
import { DatasetItem } from '../../annotator/types';

import classes from './delete-media-item.module.scss';

type DeleteMediaItemProps = {
    item: DatasetItem;
};

export const DeleteMediaItem = ({ item }: DeleteMediaItemProps) => {
    const project_id = useProjectIdentifier();
    const processingToastId = useRef<null | string | number>(null);

    const remove = $api.useMutation('delete', `/api/projects/{project_id}/dataset/items/{dataset_item_id}`, {
        onSuccess: () => {
            toast({ type: 'success', message: `Item "${item.id}" was deleted successfully`, duration: 3000 });
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

    return (
        <DialogTrigger>
            <ActionButton
                isQuiet
                aria-label='delete media item'
                isDisabled={remove.isPending}
                UNSAFE_className={classes.deleteButton}
            >
                <Delete />
            </ActionButton>
            <AlertDialog
                title='Delete Item'
                variant='confirmation'
                primaryActionLabel='Confirm'
                secondaryActionLabel='Close'
                onPrimaryAction={() => {
                    remove.mutate({ params: { path: { project_id, dataset_item_id: String(item.id) } } });
                    processingToastId.current = toast({
                        type: 'info',
                        message: `Deleting item "${item.id}"...`,
                    });
                }}
            >
                {`Are you sure you want to delete the item "${item.id}"?`}
            </AlertDialog>
        </DialogTrigger>
    );
};
