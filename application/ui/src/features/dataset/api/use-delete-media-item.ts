// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@geti/ui';
import { useOverlayTriggerState } from '@react-stately/overlays';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isFunction } from 'lodash-es';

import { $api } from '../../../api/client';

const useDeleteMediaItemMutation = (projectId: string) => {
    return $api.useMutation('delete', `/api/projects/{project_id}/dataset/media/{media_id}`, {
        meta: {
            invalidateQueries: [
                [
                    'get',
                    '/api/projects/{project_id}/dataset/media',
                    {
                        params: {
                            path: {
                                project_id: projectId,
                            },
                        },
                    },
                ],
            ],
        },
    });
};

const isFulfilled = (response: PromiseSettledResult<{ itemId: string }>) => response.status === 'fulfilled';

export const useDeleteMediaItem = () => {
    const projectId = useProjectIdentifier();
    const deleteMutation = useDeleteMediaItemMutation(projectId);

    const alertDialogState = useOverlayTriggerState({});

    const handleDeleteItems = async (ids: string[], onDeleted?: (ids: string[]) => void) => {
        alertDialogState.close();

        toast({ id: 'deleting-notification', type: 'info', message: `Deleting items...` });

        const deleteItemPromises = ids.map(async (media_id) => {
            await deleteMutation.mutateAsync({ params: { path: { project_id: projectId, media_id } } });

            return { itemId: media_id };
        });

        const responses = await Promise.allSettled(deleteItemPromises);
        const deletedIds = responses.filter(isFulfilled).map(({ value }) => value.itemId);

        isFunction(onDeleted) && onDeleted(deletedIds);

        if (deletedIds.length === 0) {
            toast({
                id: 'deleting-notification',
                type: 'error',
                message: 'Failed to delete the selected item(s). Please try again.',
            });
        } else {
            toast({
                id: 'deleting-notification',
                type: 'success',
                message: `${deletedIds.length} item(s) deleted successfully`,
                duration: 3000,
            });
        }
    };

    return {
        deleteMedia: handleDeleteItems,
        isPending: deleteMutation.isPending,
        isDeleteDialogOpen: alertDialogState.isOpen,
        openDeleteDialog: alertDialogState.open,
        closeDeleteDialog: alertDialogState.close,
    };
};
