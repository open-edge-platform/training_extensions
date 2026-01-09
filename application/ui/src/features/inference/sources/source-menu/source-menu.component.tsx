// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Item, Key, Menu, MenuTrigger, toast } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../api/client';

export interface SourceMenuProps {
    id: string;
    name: string;
    isConnected: boolean;
    onEdit: () => void;
}

export const SourceMenu = ({ id, name, isConnected, onEdit }: SourceMenuProps) => {
    const project_id = useProjectIdentifier();

    const handleOnAction = (option: Key) => {
        switch (option) {
            case 'connect':
                handleConnect();
                break;
            case 'remove':
                handleDelete();
                break;
            default:
                onEdit();
                break;
        }
    };

    const updatePipeline = $api.useMutation('patch', '/api/projects/{project_id}/pipeline', {
        meta: {
            invalidateQueries: [
                ['get', '/api/sources'],
                ['get', '/api/projects/{project_id}/pipeline'],
            ],
        },
    });

    const handleConnect = async () => {
        try {
            await updatePipeline.mutateAsync({
                params: { path: { project_id } },
                body: { source_id: id },
            });

            toast({
                type: 'success',
                message: `Successfully connected to "${name}".`,
            });
        } catch (_error) {
            toast({
                type: 'error',
                message: `Failed to connect to "${name}".`,
            });
        }
    };

    const removeSource = $api.useMutation('delete', '/api/sources/{source_id}', {
        meta: {
            invalidateQueries: [['get', '/api/sources']],
        },
    });

    const handleDelete = async () => {
        try {
            await removeSource.mutateAsync({ params: { path: { source_id: id } } });

            toast({
                type: 'success',
                message: `${name} has been removed successfully!`,
            });
        } catch (_error) {
            toast({
                type: 'error',
                message: `Failed to remove "${name}".`,
            });
        }
    };

    return (
        <MenuTrigger>
            <ActionButton isQuiet aria-label='source menu'>
                <MoreMenu />
            </ActionButton>
            <Menu onAction={handleOnAction} disabledKeys={isConnected ? ['connect', 'remove'] : []}>
                <Item key='connect'>Connect</Item>
                <Item key='edit'>Edit</Item>
                <Item key='remove'>Remove</Item>
            </Menu>
        </MenuTrigger>
    );
};
