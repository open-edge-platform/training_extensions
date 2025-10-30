// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key, useEffect, useRef, useState } from 'react';

import {
    ActionMenu,
    AlertDialog,
    DialogContainer,
    Flex,
    Item,
    PhotoPlaceholder,
    Text,
    TextField,
    type TextFieldRef,
} from '@geti/ui';
import { useNavigate } from 'react-router';
import type { SchemaProjectView } from 'src/api/openapi-spec';

import { paths } from '../../../constants/paths';

import styles from './project-list-item.module.scss';

interface ProjectEditionProps {
    onBlur: (newName: string) => void;
    name: string;
}

const ProjectEdition = ({ name, onBlur }: ProjectEditionProps) => {
    const textFieldRef = useRef<TextFieldRef>(null);
    const [newName, setNewName] = useState<string>(name);

    const handleBlur = () => {
        onBlur(newName);
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            onBlur(newName);
        } else if (e.key === 'Escape') {
            e.preventDefault();
            setNewName(name);
            onBlur(name);
        }
    };

    useEffect(() => {
        textFieldRef.current?.select();
    }, []);

    return (
        <TextField
            isQuiet
            ref={textFieldRef}
            value={newName}
            onBlur={handleBlur}
            onKeyDown={handleKeyDown}
            onChange={setNewName}
            aria-label='Edit project name'
        />
    );
};

const PROJECT_ACTIONS = {
    RENAME: 'Rename',
    DELETE: 'Delete',
};

interface ProjectActionsProps {
    onAction: (key: Key) => void;
}

interface DeleteProjectDialogProps {
    onDelete: () => void;
    projectName: string;
}

const DeleteProjectDialog = ({ projectName, onDelete }: DeleteProjectDialogProps) => {
    return (
        <AlertDialog
            title='Delete'
            variant='destructive'
            primaryActionLabel='Delete'
            onPrimaryAction={onDelete}
            cancelLabel={'Cancel'}
        >
            {`Are you sure you want to delete project "${projectName}"?`}
        </AlertDialog>
    );
};

const ProjectActions = ({ onAction }: ProjectActionsProps) => {
    return (
        <ActionMenu isQuiet onAction={onAction} aria-label={'Project actions'} UNSAFE_className={styles.actionMenu}>
            {[PROJECT_ACTIONS.RENAME, PROJECT_ACTIONS.DELETE].map((action) => (
                <Item key={action}>{action}</Item>
            ))}
        </ActionMenu>
    );
};

interface ProjectListItemProps {
    project: SchemaProjectView;
    isInEditMode: boolean;
    onBlur: (projectId: string, newName: string) => void;
    onRename: (projectId: string) => void;
    onDelete: (projectId: string) => void;
}

export const ProjectListItem = ({ project, isInEditMode, onBlur, onRename, onDelete }: ProjectListItemProps) => {
    const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState<boolean>(false);
    const navigate = useNavigate();

    const handleAction = (key: Key) => {
        if (key === PROJECT_ACTIONS.RENAME) {
            onRename(project.id);
        } else if (key === PROJECT_ACTIONS.DELETE) {
            setIsDeleteDialogOpen(true);
        }
    };

    const handleBlur = (projectId: string) => (newName: string) => {
        onBlur(projectId, newName);
    };

    const handleDelete = () => {
        onDelete(project.id);
    };

    const handleNavigateToProject = () => {
        navigate(paths.project.details({ projectId: project.id }));
    };

    return (
        <>
            <li className={styles.projectListItem} onClick={isInEditMode ? undefined : handleNavigateToProject}>
                <Flex justifyContent='space-between' alignItems='center' marginX={'size-200'}>
                    {isInEditMode ? (
                        <ProjectEdition name={project.name} onBlur={handleBlur(project.id)} />
                    ) : (
                        <Flex alignItems={'center'} gap={'size-100'}>
                            <PhotoPlaceholder
                                name={project.name}
                                indicator={project.id ?? project.name}
                                height={'size-300'}
                                width={'size-300'}
                            />
                            <Text>{project.name}</Text>
                        </Flex>
                    )}
                    <ProjectActions onAction={handleAction} />
                </Flex>
            </li>
            <DialogContainer onDismiss={() => setIsDeleteDialogOpen(false)}>
                {isDeleteDialogOpen && <DeleteProjectDialog onDelete={handleDelete} projectName={project.name} />}
            </DialogContainer>
        </>
    );
};
