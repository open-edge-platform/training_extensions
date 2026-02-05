// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { ActionButton, AlertDialog, DialogContainer, Item, Key, Menu, MenuTrigger } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import type { Model } from '../../../../../constants/shared-types';
import { usePatchPipeline } from '../../../../../hooks/api/pipeline.hook';
import { useDeleteModel } from '../../../hooks/api/use-delete-model.hook';
import { useRenameModel } from '../../../hooks/api/use-rename-model.hook';
import { isFailedModel } from '../../utils/utils';
import { RenameModelDialog } from '../model-row/rename-model-dialog.component';

const MODEL_ACTIONS = {
    ACTIVE: 'active',
    RENAME: 'rename',
    DELETE: 'delete',
};

type ModelActionsProps = {
    model: Model;
};

export const ModelActions = ({ model }: ModelActionsProps) => {
    const projectId = useProjectIdentifier();
    const deleteModelMutation = useDeleteModel();
    const renameModelMutation = useRenameModel();
    const patchPipelineMutation = usePatchPipeline();

    const [isRenameDialogOpen, setIsRenameDialogOpen] = useState(false);
    const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);

    const disabled_keys = isFailedModel(model) ? [MODEL_ACTIONS.ACTIVE, MODEL_ACTIONS.RENAME] : [];

    const handleAction = (key: Key) => {
        if (key === MODEL_ACTIONS.ACTIVE) {
            patchPipelineMutation.mutate({
                params: { path: { project_id: projectId } },
                body: { model_id: model.id },
            });
        } else if (key === MODEL_ACTIONS.DELETE) {
            setIsDeleteDialogOpen(true);
        } else if (key === MODEL_ACTIONS.RENAME) {
            setIsRenameDialogOpen(true);
        }
    };

    const handleRename = (newName: string) => {
        renameModelMutation.mutate(
            {
                params: { path: { project_id: projectId, model_id: model.id } },
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
        deleteModelMutation.mutate({ params: { path: { project_id: projectId, model_id: model.id } } });
    };

    return (
        <>
            <MenuTrigger>
                <ActionButton isQuiet aria-label={'Model actions'}>
                    <MoreMenu />
                </ActionButton>
                <Menu onAction={handleAction} aria-label={'Model actions menu'} disabledKeys={disabled_keys}>
                    <Item key={MODEL_ACTIONS.ACTIVE}>Set as active</Item>
                    <Item key={MODEL_ACTIONS.RENAME}>Rename</Item>
                    <Item key={MODEL_ACTIONS.DELETE}>Delete</Item>
                </Menu>
            </MenuTrigger>

            <DialogContainer onDismiss={() => setIsRenameDialogOpen(false)}>
                {isRenameDialogOpen && (
                    <RenameModelDialog
                        currentName={model.name ?? ''}
                        onRename={handleRename}
                        isPending={renameModelMutation.isPending}
                        onClose={() => setIsRenameDialogOpen(false)}
                    />
                )}
            </DialogContainer>
            <DialogContainer onDismiss={() => setIsDeleteDialogOpen(false)}>
                {isDeleteDialogOpen && (
                    <AlertDialog
                        title='Delete model'
                        variant='destructive'
                        primaryActionLabel='Delete'
                        onPrimaryAction={handleDelete}
                        cancelLabel='Cancel'
                    >
                        {`Are you sure you want to delete model "${model.name ?? 'Unnamed Model'}"?`}
                    </AlertDialog>
                )}
            </DialogContainer>
        </>
    );
};
