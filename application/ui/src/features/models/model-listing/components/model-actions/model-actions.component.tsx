// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { ActionButton, AlertDialog, DialogContainer, Item, Key, Menu, MenuTrigger } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import type { Model } from '../../../../../constants/shared-types';
import { usePatchPipeline } from '../../../../../hooks/api/pipeline.hook';
import { useDeleteModel } from '../../../hooks/api/use-delete-model.hook';
import { useRenameModel } from '../../../hooks/api/use-rename-model.hook';
import { TrainingLogsDialog } from '../../../training-logs/training-logs-dialog.component';
import { isFailedModel, isTrainingModel } from '../../utils/utils';
import { RenameModelDialog } from '../model-row/rename-model-dialog.component';

const MODEL_ACTIONS = {
    ACTIVATE: 'activate',
    RENAME: 'rename',
    DELETE: 'delete',
    VIEW_LOGS: 'view_logs',
};

enum DIALOG_TYPES {
    RENAME = 'rename',
    DELETE = 'delete',
    LOGS = 'logs',
}
type ModelActionsProps = {
    model: Model;
};

export const ModelActions = ({ model }: ModelActionsProps) => {
    const projectId = useProjectIdentifier();
    const deleteModelMutation = useDeleteModel();
    const renameModelMutation = useRenameModel();
    const patchPipelineMutation = usePatchPipeline();

    const [isDialogOpen, setIsDialogOpen] = useState<DIALOG_TYPES | null>(null);

    const disableRenameAndActive = isFailedModel(model) || isTrainingModel(model);
    const disabledKeys = [];
    if (disableRenameAndActive) disabledKeys.push(MODEL_ACTIONS.ACTIVATE, MODEL_ACTIONS.RENAME);
    if (isTrainingModel(model)) disabledKeys.push(MODEL_ACTIONS.VIEW_LOGS);

    const handleAction = (key: Key) => {
        if (key === MODEL_ACTIONS.ACTIVATE) {
            patchPipelineMutation.mutate({
                params: { path: { project_id: projectId } },
                body: { model_id: model.id },
            });
        } else if (key === MODEL_ACTIONS.DELETE) {
            setIsDialogOpen(DIALOG_TYPES.DELETE);
        } else if (key === MODEL_ACTIONS.RENAME) {
            setIsDialogOpen(DIALOG_TYPES.RENAME);
        } else if (key === MODEL_ACTIONS.VIEW_LOGS) {
            setIsDialogOpen(DIALOG_TYPES.LOGS);
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
                    setIsDialogOpen(null);
                },
            }
        );
    };

    const handleDelete = () => {
        deleteModelMutation.mutate({
            params: {
                path: { project_id: projectId, model_id: model.id },
                query: { files_only: true },
            },
        });
    };

    return (
        <>
            <MenuTrigger>
                <ActionButton isQuiet aria-label={'Model actions'}>
                    <MoreMenu />
                </ActionButton>
                <Menu onAction={handleAction} aria-label={'Model actions menu'} disabledKeys={disabledKeys}>
                    <Item key={MODEL_ACTIONS.ACTIVATE}>Set as active</Item>
                    <Item key={MODEL_ACTIONS.RENAME}>Rename</Item>
                    <Item key={MODEL_ACTIONS.VIEW_LOGS}>View training logs</Item>
                    <Item key={MODEL_ACTIONS.DELETE}>Delete</Item>
                </Menu>
            </MenuTrigger>

            <DialogContainer onDismiss={() => setIsDialogOpen(null)}>
                {isDialogOpen === DIALOG_TYPES.RENAME && (
                    <RenameModelDialog
                        currentName={model.name ?? ''}
                        onRename={handleRename}
                        isPending={renameModelMutation.isPending}
                        onClose={() => setIsDialogOpen(null)}
                    />
                )}
            </DialogContainer>
            <DialogContainer onDismiss={() => setIsDialogOpen(null)}>
                {isDialogOpen === DIALOG_TYPES.DELETE && (
                    <AlertDialog
                        title='Delete model files'
                        variant='destructive'
                        primaryActionLabel='Delete files'
                        onPrimaryAction={handleDelete}
                        cancelLabel='Cancel'
                    >
                        {`Are you sure you want to delete files for model "${model.name ?? 'Unnamed Model'}"?`}
                    </AlertDialog>
                )}
            </DialogContainer>
            <DialogContainer type={'fullscreen'} onDismiss={() => setIsDialogOpen(null)}>
                {isDialogOpen === DIALOG_TYPES.LOGS && <TrainingLogsDialog modelId={model.id} />}
            </DialogContainer>
        </>
    );
};
