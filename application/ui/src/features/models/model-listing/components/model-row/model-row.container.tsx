// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { AlertDialog, DialogContainer, Key } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import type { Model } from '../../../../../constants/shared-types';
import { useDeleteModel, useGetModel, useRenameModel } from '../../../../../hooks/api/models.hook';
import { usePatchPipeline } from '../../../../../hooks/api/pipeline.hook';
import { useModelListing } from '../../provider/model-listing-provider';
import { ModelRow } from './model-row.component';
import { RenameModelDialog } from './rename-model-dialog.component';

const MODEL_ACTIONS = {
    ACTIVE: 'active',
    RENAME: 'rename',
    DELETE: 'delete',
    EXPORT: 'export',
};

type ModelRowContainerProps = {
    model: Model;
};

export const ModelRowContainer = ({ model }: ModelRowContainerProps) => {
    const projectId = useProjectIdentifier();
    const { activeModelArchitectureId, onExpandModel } = useModelListing();
    const { data: parentRevisionModel } = useGetModel(model.parent_revision);
    const deleteModelMutation = useDeleteModel();
    const renameModelMutation = useRenameModel();
    const patchPipelineMutation = usePatchPipeline();

    const [isRenameDialogOpen, setIsRenameDialogOpen] = useState(false);
    const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);

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
            <ModelRow
                model={model}
                activeModelArchitectureId={activeModelArchitectureId}
                parentRevisionModel={parentRevisionModel}
                onExpandModel={onExpandModel}
                onModelAction={handleAction}
            />
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
                        title='Delete'
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
