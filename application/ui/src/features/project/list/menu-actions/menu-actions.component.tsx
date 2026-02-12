// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { CSSProperties } from 'react';

import { ActionButton, Item, Key, Menu, MenuTrigger } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';
import { useOverlayTriggerState } from 'react-stately';

import { DeleteProjectDialog } from './delete-project-dialog/delete-project-dialog.component';
import { EditProjectNameDialog } from './edit-project-name-dialog/edit-project-name-dialog.component';

type MenuActionsProps = {
    projectId: string;
    projectName: string;
    actionButtonStyle: CSSProperties;
};

export const MenuActions = ({ projectId, projectName, actionButtonStyle }: MenuActionsProps) => {
    const deleteProjectDialogState = useOverlayTriggerState({});
    const editProjectNameDialogState = useOverlayTriggerState({});

    const handleMenuAction = (key: Key) => {
        switch (key) {
            case 'rename':
                editProjectNameDialogState.open();
                break;
            case 'delete':
                deleteProjectDialogState.open();
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
                isOpen={editProjectNameDialogState.isOpen}
                onClose={editProjectNameDialogState.close}
            />

            <DeleteProjectDialog
                projectId={projectId}
                projectName={projectName}
                isOpen={deleteProjectDialogState.isOpen}
                onClose={deleteProjectDialogState.close}
            />
        </>
    );
};
