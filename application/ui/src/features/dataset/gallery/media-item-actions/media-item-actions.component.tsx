// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key } from 'react';

import { ActionButton, DialogContainer, Item, Menu, MenuTrigger } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';

import { useDeleteMediaItem } from '../../api/use-delete-media-item';
import { AlertDialogContent } from '../delete-media-item/alert-dialog-content.component';

const MEDIA_ACTIONS = {
    DOWNLOAD: 'download',
    DELETE: 'delete',
};

type MediaItemActionsProps = {
    id: string;
    mediaUrl: string;
    mediaFileName: string;
    onDeleted: (deletedIds: string[]) => void;
};

export const MediaItemActions = ({ id, onDeleted, mediaUrl, mediaFileName }: MediaItemActionsProps) => {
    const { closeDeleteDialog, openDeleteDialog, isDeleteDialogOpen, deleteMedia, isPending } = useDeleteMediaItem();

    const handleDownload = () => {
        const link = document.createElement('a');
        link.href = mediaUrl;
        // Just in case, the "Content-Disposition" header takes precedence over this filename.
        link.download = mediaFileName;
        link.hidden = true;
        link.click();
    };

    const handleAction = (key: Key) => {
        if (key === MEDIA_ACTIONS.DOWNLOAD) {
            handleDownload();
        } else if (key === MEDIA_ACTIONS.DELETE) {
            openDeleteDialog();
        }
    };

    const handleDeleteMedia = async () => {
        await deleteMedia([id], onDeleted);
    };

    return (
        <>
            <MenuTrigger>
                <ActionButton isQuiet aria-label={'Media actions'} isDisabled={isPending}>
                    <MoreMenu />
                </ActionButton>
                <Menu onAction={handleAction} aria-label={'Media actions menu'}>
                    <Item key={MEDIA_ACTIONS.DOWNLOAD}>Download</Item>
                    <Item key={MEDIA_ACTIONS.DELETE}>Delete</Item>
                </Menu>
            </MenuTrigger>
            <DialogContainer onDismiss={closeDeleteDialog}>
                {isDeleteDialogOpen && <AlertDialogContent itemsIds={[id]} onPrimaryAction={handleDeleteMedia} />}
            </DialogContainer>
        </>
    );
};
