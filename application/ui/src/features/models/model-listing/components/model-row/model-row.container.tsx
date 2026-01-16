// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { SchemaModelView } from '../../../../../api/openapi-spec';
import { useDeleteModel } from '../../../hooks/api/use-delete-model.hook';
import { useGetModel } from '../../../hooks/api/use-get-model.hook';
import { useModelListing } from '../../provider/model-listing-provider';
import { ModelRow } from './model-row.component';

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
    const deleteModelMutation = useDeleteModel(model.id);

    const handleAction = (key: Key) => {
        if (key === MODEL_ACTIONS.DELETE) {
            deleteModelMutation?.mutate({ params: { path: { project_id: projectId, model_id: model.id } } });
        } else if (key === MODEL_ACTIONS.RENAME) {
        } else if (key === MODEL_ACTIONS.EXPORT) {
            // TODO: Implement export functionality
        }
    };

    return (
        <ModelRow
            model={model}
            activeModelId={activeModelId}
            parentRevisionModel={parentRevisionModel?.data}
            onExpandModel={onExpandModel}
            handleModelAction={handleAction}
        />
    );
};
