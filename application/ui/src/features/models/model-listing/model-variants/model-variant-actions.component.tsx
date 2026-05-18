// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key } from 'react';

import { ActionButton, Item, Menu, MenuTrigger } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';

import { ModelVariant } from '../../../../constants/shared-types';

const MODEL_VARIANT_ACTIONS = {
    DOWNLOAD: 'download',
};

type ModelVariantActionsProps = {
    modelVariant: ModelVariant;
    onDownload: (modelVariantId: string) => void;
};

export const ModelVariantActions = ({ modelVariant, onDownload }: ModelVariantActionsProps) => {
    const handleAction = (key: Key) => {
        if (key === MODEL_VARIANT_ACTIONS.DOWNLOAD) {
            onDownload(modelVariant.id);
        }
    };

    return (
        <MenuTrigger>
            <ActionButton isQuiet aria-label={`Model variant actions ${modelVariant.id}`}>
                <MoreMenu />
            </ActionButton>
            <Menu onAction={handleAction} aria-label={'Model variant actions menu'}>
                <Item key={MODEL_VARIANT_ACTIONS.DOWNLOAD}>Download</Item>
            </Menu>
        </MenuTrigger>
    );
};
