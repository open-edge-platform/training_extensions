// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState, type CSSProperties } from 'react';

import { ActionButton, Item, Menu, MenuTrigger } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';
import { useOverlayTriggerState } from 'react-stately';

import { EnablePipelineBlockedDialog } from '../../../../components/enable-pipeline-blocked-dialog/enable-pipeline-blocked-dialog.component';
import { DeleteProjectDialog } from '../../../../components/project-dialogs/delete-project-dialog.component';
import { EditProjectNameDialog } from '../../../../components/project-dialogs/edit-project-name-dialog.component';
import { useProjectMenuActions } from './use-project-menu-actions';

import classes from './menu-actions.module.scss';

type MenuActionsProps = {
    projectId: string;
    projectName: string;
    isPipelineRunning?: boolean;
    actionButtonStyle?: CSSProperties;
    onDeleted?: () => void;
    projectsNames: string[];
};

export const MenuActions = ({
    projectId,
    projectName,
    isPipelineRunning,
    actionButtonStyle,
    onDeleted,
    projectsNames,
}: MenuActionsProps) => {
    const [isEnableBlockedDialogOpen, setIsEnableBlockedDialogOpen] = useState(false);
    const deleteProjectDialogState = useOverlayTriggerState({});
    const editProjectNameDialogState = useOverlayTriggerState({});

    const { menuActions, handleAction } = useProjectMenuActions(
        projectId,
        {
            onRename: editProjectNameDialogState.open,
            onDelete: deleteProjectDialogState.open,
            onEnableBlocked: () => setIsEnableBlockedDialogOpen(true),
        },
        isPipelineRunning
    );

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
                    {menuActions.map(({ key, label }) => (
                        <Item key={key}>{label}</Item>
                    ))}
                </Menu>
            </MenuTrigger>

            <EnablePipelineBlockedDialog
                isOpen={isEnableBlockedDialogOpen}
                onClose={() => setIsEnableBlockedDialogOpen(false)}
            />

            <EditProjectNameDialog
                projectId={projectId}
                projectName={projectName}
                projectsNames={projectsNames}
                isOpen={editProjectNameDialogState.isOpen}
                onClose={editProjectNameDialogState.close}
            />

            <DeleteProjectDialog
                projectId={projectId}
                projectName={projectName}
                isOpen={deleteProjectDialogState.isOpen}
                onClose={deleteProjectDialogState.close}
                onDeleted={onDeleted}
            />
        </>
    );
};
