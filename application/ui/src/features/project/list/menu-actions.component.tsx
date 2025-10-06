// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Item, Key, Menu, MenuTrigger, toast } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';
import { useNavigate } from 'react-router';

import { $api } from '../../../api/client';
import { paths } from '../../../constants/paths';

export const MenuActions = ({ projectId }: { projectId: string }) => {
    const navigate = useNavigate();

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
            case 'edit':
                navigate(paths.project.details({ projectId }));
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
            <ActionButton
                isQuiet
                UNSAFE_style={{ fill: 'var(--spectrum-gray-900)' }}
                aria-label={'open project options'}
                data-testid={projectId}
            >
                <MoreMenu />
            </ActionButton>
            <Menu onAction={handleMenuAction}>
                <Item key={'edit'}>Edit</Item>
                <Item key={'delete'}>Delete</Item>
            </Menu>
        </MenuTrigger>
    );
};
