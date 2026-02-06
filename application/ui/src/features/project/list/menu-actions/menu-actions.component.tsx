// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CSSProperties } from 'react';

import { ActionButton, Item, Key, Menu, MenuTrigger, toast } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';
import { useOverlayTriggerState } from 'react-stately';

import { useDeleteProject } from '../../../../hooks/api/project.hook';
import { EditProjectNameDialog } from './edit-project-name-dialog.component';

type MenuActionsProps = {
    projectId: string;
    projectName: string;
    actionButtonStyle: CSSProperties;
};

export const MenuActions = ({ projectId, projectName, actionButtonStyle }: MenuActionsProps) => {
    const deleteMutation = useDeleteProject();
    const editProjectNameDialogState = useOverlayTriggerState({});

    const handleMenuAction = (key: Key) => {
        switch (key) {
            case 'rename':
                editProjectNameDialogState.open();
                break;
            case 'delete':
                deleteMutation.mutate(
                    { params: { path: { project_id: projectId } } },
                    {
                        onSuccess: () => {
                            toast({ type: 'success', message: 'Project deleted successfully' });
                        },
                        onError: () => {
                            toast({ type: 'error', message: 'Failed to delete project' });
                        },
                    }
                );
                break;
            default:
                break;
        }
    };

    return (
        <>
            <MenuTrigger>
                <ActionButton
                    isQuiet
                    UNSAFE_style={{
                        fill: 'var(--spectrum-gray-900)',
                        ...actionButtonStyle,
                    }}
                    aria-label={'open project options'}
                    data-testid={projectId}
                >
                    <MoreMenu />
                </ActionButton>
                <Menu onAction={handleMenuAction}>
                    <Item key={'rename'}>Rename</Item>
                    <Item key={'delete'}>Delete</Item>
                </Menu>
            </MenuTrigger>

            <EditProjectNameDialog
                projectId={projectId}
                projectName={projectName}
                onClose={editProjectNameDialogState.close}
                isOpen={editProjectNameDialogState.isOpen}
            />
        </>
    );
};
