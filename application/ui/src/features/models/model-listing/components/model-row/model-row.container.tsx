// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { AlertDialog, DialogContainer, Key } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { SchemaModelView } from '../../../../../api/openapi-spec';
import { useDeleteModel } from '../../../hooks/api/use-delete-model.hook';
import { useGetModel } from '../../../hooks/api/use-get-model.hook';
import { useRenameModel } from '../../../hooks/api/use-rename-model.hook';
import { useModelListing } from '../../provider/model-listing-provider';
import { ModelRow } from './model-row.component';
import { RenameModelDialog } from './rename-model-dialog.component';

const MODEL_ACTIONS = {
    RENAME: 'rename',
    DELETE: 'delete',
    EXPORT: 'export',
};

type ModelRowContainerProps = {
    model: SchemaModelView;
};

export const ModelRowContainer = ({ model }: ModelRowContainerProps) => {
    const projectId = useProjectIdentifier();
    const { activeModelId, onExpandModel } = useModelListing();
    const parentRevisionModel = useGetModel(model.parent_revision);
    const deleteModelMutation = useDeleteModel();
    const renameModelMutation = useRenameModel();
    const [isRenameDialogOpen, setIsRenameDialogOpen] = useState(false);
    const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);

    const handleAction = (key: Key) => {
        if (key === MODEL_ACTIONS.DELETE) {
            setIsDeleteDialogOpen(true);
        } else if (key === MODEL_ACTIONS.RENAME) {
            setIsRenameDialogOpen(true);
        } else if (key === MODEL_ACTIONS.EXPORT) {
            // TODO: Implement export functionality
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
                activeModelId={activeModelId}
                parentRevisionModel={parentRevisionModel?.data}
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
