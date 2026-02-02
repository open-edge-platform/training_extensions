// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { ActionButton, AlertDialog, DialogContainer, Item, Key, Menu, MenuTrigger } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { useDeleteDatasetRevision } from '../../hooks/use-delete-dataset-revision.hook';
import { useRenameDatasetRevision } from '../../hooks/use-rename-dataset-revision.hook';
import type { DatasetGroup } from '../../types';
import { RenameDatasetRevisionDialog } from '../group-headers/rename-dataset-revision-dialog.component';

type DatasetActionsProps = {
    dataset: DatasetGroup;
};

export const DatasetActions = ({ dataset }: DatasetActionsProps) => {
    const projectId = useProjectIdentifier();
    const renameDatasetRevisionMutation = useRenameDatasetRevision();
    const deleteDatasetRevisionMutation = useDeleteDatasetRevision();

    const [isRenameDialogOpen, setIsRenameDialogOpen] = useState(false);
    const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);

    const handleDatasetMenuAction = (key: Key) => {
        switch (key) {
            case 'rename':
                setIsRenameDialogOpen(true);
                break;
            case 'delete':
                setIsDeleteDialogOpen(true);
                break;
            default:
                break;
        }
    };

    const handleRename = (newName: string) => {
        renameDatasetRevisionMutation.mutate(
            {
                params: { path: { project_id: projectId, dataset_revision_id: dataset.id } },
                body: { name: newName },
            },
            {
                onSuccess: () => {
                    setIsRenameDialogOpen(false);
                },
            }
        );
    };

    const handleDelete = () => {
        deleteDatasetRevisionMutation.mutate({
            params: { path: { project_id: projectId, dataset_revision_id: dataset.id } },
        });
    };

    return (
        <>
            <MenuTrigger>
                <ActionButton isQuiet aria-label={'Dataset actions'}>
                    <MoreMenu />
                </ActionButton>
                <Menu onAction={handleDatasetMenuAction} aria-label={'Dataset actions menu'}>
                    <Item key={'rename'}>Rename</Item>
                    <Item key={'delete'}>Delete</Item>
                </Menu>
            </MenuTrigger>

            <DialogContainer onDismiss={() => setIsRenameDialogOpen(false)}>
                {isRenameDialogOpen && (
                    <RenameDatasetRevisionDialog
                        currentName={dataset.name}
                        onRename={handleRename}
                        isPending={renameDatasetRevisionMutation.isPending}
                        onClose={() => setIsRenameDialogOpen(false)}
                    />
                )}
            </DialogContainer>

            <DialogContainer onDismiss={() => setIsDeleteDialogOpen(false)}>
                {isDeleteDialogOpen && (
                    <AlertDialog
                        title='Delete dataset revision'
                        variant='destructive'
                        primaryActionLabel='Delete'
                        onPrimaryAction={handleDelete}
                        cancelLabel='Cancel'
                    >
                        {`Are you sure you want to delete dataset revision "${dataset.name}"? ` +
                            `You will still be able to see the model statistics but you won't ` +
                            `be able to access the training dataset files.`}
                    </AlertDialog>
                )}
            </DialogContainer>
        </>
    );
};
