// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { AlertDialog, DialogContainer, toast } from '@geti/ui';
import { useDeleteProject } from 'hooks/api/project.hook';

type DeleteProjectDialogProps = {
    isOpen: boolean;
    projectId: string;
    projectName: string;
    onClose: () => void;
    onSuccess?: () => void;
};

export const DeleteProjectDialog = ({
    isOpen,
    projectId,
    projectName,
    onClose,
    onSuccess,
}: DeleteProjectDialogProps) => {
    const deleteMutation = useDeleteProject();

    const handleDelete = () => {
        deleteMutation.mutate(
            { params: { path: { project_id: projectId } } },
            {
                onSuccess: () => {
                    onClose();
                    toast({ type: 'success', message: 'Project deleted successfully' });
                    onSuccess?.();
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
