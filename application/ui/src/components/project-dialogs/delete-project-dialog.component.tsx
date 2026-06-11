// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { AlertDialog, DialogContainer } from '@geti/ui';
import { useDeleteProject } from 'hooks/api/project.hook';

import { toast } from '../toast/toast.component';

type DeleteProjectDialogProps = {
    isOpen: boolean;
    projectId: string;
    projectName: string;
    onClose: () => void;
    onDeleted?: () => void;
};

export const DeleteProjectDialog = ({
    isOpen,
    projectId,
    projectName,
    onClose,
    onDeleted,
}: DeleteProjectDialogProps) => {
    const deleteMutation = useDeleteProject();

    const handleDelete = () => {
        deleteMutation.mutate(
            { params: { path: { project_id: projectId } } },
            {
                onSuccess: () => {
                    onClose();
                    onDeleted?.();
                    toast({ type: 'success', message: 'Project deleted successfully' });
                },
            }
        );
    };

    return (
        <DialogContainer onDismiss={onClose}>
            {isOpen && (
                <AlertDialog
                    title='Delete'
                    variant='destructive'
                    cancelLabel='Cancel'
                    primaryActionLabel='Delete'
                    onPrimaryAction={handleDelete}
                    onSecondaryAction={onClose}
                    autoFocusButton='primary'
                    isPrimaryActionDisabled={deleteMutation.isPending}
                >
                    {`Are you sure you want to delete project "${projectName}"?`}
                </AlertDialog>
            )}
        </DialogContainer>
    );
};
