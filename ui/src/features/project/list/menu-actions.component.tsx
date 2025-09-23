// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Item, Key, Menu, MenuTrigger, toast } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';

import { $api } from '../../../api/client';

export const MenuActions = ({ projectId }: { projectId: string }) => {
    const deleteMutation = $api.useMutation('delete', '/api/projects/{project_id}', {
        onSuccess: () => {
            toast({ type: 'success', message: 'Project deleted successfully' });
        },
        onError: () => {
            toast({ type: 'error', message: 'Failed to delete project' });
        },
    });

    const handleMenuAction = (key: Key) => {
        switch (key) {
            case 'export':
                // Handle export action
                break;
            case 'duplicate':
                // Handle duplicate action
                break;
            case 'delete':
                deleteMutation.mutate({ params: { path: { project_id: projectId } } });
                break;
            default:
                break;
        }
    };

    return (
        <MenuTrigger>
            <ActionButton isQuiet UNSAFE_style={{ fill: 'var(--spectrum-gray-900)' }}>
                <MoreMenu />
            </ActionButton>
            <Menu onAction={handleMenuAction}>
                <Item key={'export'}>Export</Item>
                <Item key={'duplicate'}>Duplicate</Item>
                <Item key={'delete'}>Delete</Item>
            </Menu>
        </MenuTrigger>
    );
};
