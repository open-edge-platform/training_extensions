// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { CSSProperties } from 'react';

import { ActionButton, Item, Menu, MenuTrigger } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';
import { useOverlayTriggerState } from 'react-stately';

import { DeleteProjectDialog } from '../../../../components/project-dialogs/delete-project-dialog.component';
import { EditProjectNameDialog } from '../../../../components/project-dialogs/edit-project-name-dialog.component';
import { useProjectMenuActions } from './use-project-menu-actions';

import classes from './menu-actions.module.scss';

type MenuActionsProps = {
    projectId: string;
    projectName: string;
    actionButtonStyle?: CSSProperties;
};

export const MenuActions = ({ projectId, projectName, actionButtonStyle }: MenuActionsProps) => {
    const deleteProjectDialogState = useOverlayTriggerState({});
    const editProjectNameDialogState = useOverlayTriggerState({});

    const { menuActions, handleAction } = useProjectMenuActions(projectId, {
        onRename: editProjectNameDialogState.open,
        onDelete: deleteProjectDialogState.open,
    });

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
                <Menu onAction={handleAction} UNSAFE_className={classes.actionMenu}>
                    {Object.entries(menuActions).map(([key, label]) => (
                        <Item key={key}>{label}</Item>
                    ))}
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
