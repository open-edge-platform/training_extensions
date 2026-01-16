// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key } from 'react';

import { ActionButton, Item, Menu, MenuTrigger, toast } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';

import { $api } from '../../../../../api/client';
import { useProjectIdentifier } from '../../../../../hooks/use-project-identifier.hook';

export interface SinkMenuProps {
    id: string;
    name: string;
    isConnected: boolean;
    onEdit: () => void;
}

export const SinkMenu = ({ id, name, isConnected, onEdit }: SinkMenuProps) => {
    const project_id = useProjectIdentifier();
    const removeSink = $api.useMutation('delete', '/api/sinks/{sink_id}', {
        meta: {
            invalidateQueries: [['get', '/api/sinks']],
        },
    });

    const updatePipeline = $api.useMutation('patch', '/api/projects/{project_id}/pipeline', {
        meta: {
            invalidateQueries: [['get', '/api/projects/{project_id}/pipeline']],
        },
    });

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

    const handleConnect = async () => {
        try {
            await updatePipeline.mutateAsync({
                params: { path: { project_id } },
                body: { sink_id: id },
            });

            toast({
                type: 'success',
                message: `Successfully connected to "${name}"`,
            });
        } catch (_error) {
            toast({
                type: 'error',
                message: `Failed to connect to "${name}".`,
            });
        }
    };

    const handleDelete = async () => {
        try {
            await removeSink.mutateAsync({ params: { path: { sink_id: id } } });

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
            <ActionButton isQuiet aria-label='sink menu'>
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
