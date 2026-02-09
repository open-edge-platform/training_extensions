// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { AlertDialog, DialogContainer, toast } from '@geti/ui';
import { useDeleteProject } from 'hooks/api/project.hook';

interface DeleteProjectDialogProps {
    isOpen: boolean;
    projectId: string;
    projectName: string;
    onClose: () => void;
}

export const DeleteProjectDialog = ({ isOpen, projectId, projectName, onClose }: DeleteProjectDialogProps) => {
    const deleteMutation = useDeleteProject();

    const handleDelete = () => {
        deleteMutation.mutate(
            { params: { path: { project_id: projectId } } },
            {
                onSuccess: () => {
                    toast({ type: 'success', message: 'Project deleted successfully' });
                },
                onError: () => {
                    toast({ type: 'error', message: 'Failed to delete project' });
                },
                onSettled: () => {
                    onClose();
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
                    isPrimaryActionDisabled={deleteMutation.isPending}
                >
                    {`Are you sure you want to delete project "${projectName}"?`}
                </AlertDialog>
            )}
        </DialogContainer>
    );
};
