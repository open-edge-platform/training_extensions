// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key } from 'react';

import { ActionButton, Item, Menu, MenuTrigger } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';
import { usePatchPipeline } from 'hooks/api/pipeline.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { ModelVariant } from '../../../../constants/shared-types';

const MODEL_VARIANT_ACTIONS = {
    ACTIVATE: 'activate',
    DOWNLOAD: 'download',
};

type ModelVariantActionsProps = {
    modelVariant: ModelVariant;
    onDownload: (modelVariantId: string) => void;
};

export const ModelVariantActions = ({ modelVariant, onDownload }: ModelVariantActionsProps) => {
    const projectId = useProjectIdentifier();

    const patchPipelineMutation = usePatchPipeline();

    const handleAction = (key: Key) => {
        if (key === MODEL_VARIANT_ACTIONS.ACTIVATE) {
            patchPipelineMutation.mutate({
                params: { path: { project_id: projectId } },
                body: { model_id: modelVariant.id },
            });
        } else if (key === MODEL_VARIANT_ACTIONS.DOWNLOAD) {
            onDownload(modelVariant.id);
        }
    };

    return (
        <MenuTrigger>
            <ActionButton isQuiet aria-label={`Model variant actions ${modelVariant.id}`}>
                <MoreMenu />
            </ActionButton>
            <Menu onAction={handleAction} aria-label={'Model actions menu'}>
                <Item key={MODEL_VARIANT_ACTIONS.ACTIVATE}>Set as active</Item>
                <Item key={MODEL_VARIANT_ACTIONS.DOWNLOAD}>Download</Item>
            </Menu>
        </MenuTrigger>
    );
};
