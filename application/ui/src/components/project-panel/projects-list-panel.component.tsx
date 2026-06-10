// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import {
    ActionButton,
    Badge,
    ButtonGroup,
    Content,
    Dialog,
    DialogTrigger,
    dimensionValue,
    Divider,
    Flex,
    Header,
    Heading,
    Tag,
    Text,
    View,
} from '@geti/ui';
import { Edit } from '@geti/ui/icons';
import { useProjects } from 'hooks/api/project.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { partition } from 'lodash-es';
import { useNavigate } from 'react-router';
import { useOverlayTriggerState } from 'react-stately';

import { EnablePipelineBlockedDialog } from '../../components/enable-pipeline-blocked-dialog/enable-pipeline-blocked-dialog.component';
import { DeleteProjectDialog } from '../../components/project-dialogs/delete-project-dialog.component';
import { EditProjectNameDialog } from '../../components/project-dialogs/edit-project-name-dialog.component';
import { paths } from '../../constants/paths';
import {
    ProjectActionsMenu,
    type ProjectActionMetadata,
} from '../../features/project/list/menu-actions/menu-actions.component';
import { getProjectTypeTitle } from '../../features/project/list/util';
import { ProjectThumbnail } from './project-thumbnail/project-thumbnail.component';
import { ProjectsList } from './projects-list.component';

import classes from './projects-list.module.scss';

type SelectedProjectProps = {
    name: string;
    id: string | undefined;
    isActive: boolean;
};

const SelectedProjectButton = ({ name, id, isActive }: SelectedProjectProps) => {
    return (
        <ActionButton
            aria-label={`Selected project ${name}`}
            isQuiet
            height={'max-content'}
            staticColor={'white'}
            UNSAFE_className={classes.selectedProjectButton}
        >
            <View margin='size-50'>
                <ProjectThumbnail
                    // when selected project changes, we want to reset the project thumbnail
                    key={id}
                    project={{ name, id: id ?? name }}
                    height={'size-400'}
                    width={'size-400'}
                />
            </View>
            <Flex direction={'column'} minWidth={0}>
                <View paddingStart={'size-50'} width={'100%'} UNSAFE_className={classes.selectedProjectName}>
                    <span title={name}>{name}</span>
                </View>
                {isActive ? <Tag className={classes.statusTag} text={'Active'} /> : null}
            </Flex>
        </ActionButton>
    );
};

const ManageProjects = () => {
    const navigate = useNavigate();

    const navigateToProjectsList = () => {
        navigate(paths.project.index({}));
    };

    return (
        <ActionButton
            isQuiet
            width={'100%'}
            UNSAFE_className={classes.manageProjectsButton}
            onPress={navigateToProjectsList}
        >
            <Edit />
            <Text>Manage projects</Text>
        </ActionButton>
    );
};

export const ProjectsListPanel = () => {
    const navigate = useNavigate();
    const projectId = useProjectIdentifier();
    const { data } = useProjects();
    const projectSelectorState = useOverlayTriggerState({});
    const deleteProjectDialogState = useOverlayTriggerState({});
    const editProjectNameDialogState = useOverlayTriggerState({});
    const [isEnableBlockedDialogOpen, setIsEnableBlockedDialogOpen] = useState(false);
    const [projectActionMetadata, setProjectActionMetadata] = useState<ProjectActionMetadata | null>(null);

    const [[selectedProject], otherProjects] = partition(data, (project) => project.id === projectId);
    const selectedProjectName = selectedProject?.name ?? '';
    const hasActivePipeline = Boolean(selectedProject?.active_pipeline);

    const otherProjectNames = otherProjects.map(({ name }) => name);

    const taskType = getProjectTypeTitle(selectedProject?.task);

    const handleDeleted = () => {
        if (selectedProject.id === projectActionMetadata?.projectId) {
            navigate(paths.project.index({}));
        }

        setProjectActionMetadata(null);
    };

    const handleRename = (metadata: ProjectActionMetadata) => {
        setProjectActionMetadata(metadata);
        projectSelectorState.close();
        editProjectNameDialogState.open();
    };

    const handleDelete = (metadata: ProjectActionMetadata) => {
        setProjectActionMetadata(metadata);
        projectSelectorState.close();
        deleteProjectDialogState.open();
    };

    const handleEnableBlocked = (metadata: ProjectActionMetadata) => {
        setProjectActionMetadata(metadata);
        projectSelectorState.close();
        setIsEnableBlockedDialogOpen(true);
    };

    const handleCloseEnableBlocked = () => {
        setIsEnableBlockedDialogOpen(false);
        setProjectActionMetadata(null);
    };

    const handleCloseEdit = () => {
        editProjectNameDialogState.close();
        setProjectActionMetadata(null);
    };

    return (
        <>
            <DialogTrigger
                type='popover'
                hideArrow
                isOpen={projectSelectorState.isOpen}
                onOpenChange={projectSelectorState.setOpen}
            >
                <SelectedProjectButton name={selectedProjectName} id={projectId} isActive={hasActivePipeline} />

                <Dialog width={'size-4600'} UNSAFE_className={classes.dialog}>
                    {selectedProject !== undefined && (
                        <Header>
                            <Flex
                                direction={'column'}
                                justifyContent={'center'}
                                width={'100%'}
                                alignItems={'center'}
                                UNSAFE_style={{
                                    padding: 'var(--spectrum-global-dimension-size-200)',
                                }}
                                gap={'size-100'}
                            >
                                <ProjectThumbnail
                                    // when selected project changes, we want to reset the project thumbnail
                                    key={selectedProject.id}
                                    project={selectedProject}
                                    height={'size-1000'}
                                    width={'size-1000'}
                                />
                                <View width={'100%'} position={'relative'}>
                                    <Flex direction={'column'} alignItems={'center'} gap={'size-50'}>
                                        <Heading
                                            UNSAFE_className={classes.dialogProjectName}
                                            level={2}
                                            marginBottom={0}
                                        >
                                            {selectedProjectName}
                                        </Heading>

                                        {taskType !== undefined && (
                                            <Badge variant={'neutral'}>
                                                <Text>{taskType}</Text>
                                            </Badge>
                                        )}
                                    </Flex>

                                    <ProjectActionsMenu
                                        projectId={selectedProject.id}
                                        projectName={selectedProject.name}
                                        isPipelineRunning={selectedProject.active_pipeline}
                                        projectNames={otherProjectNames}
                                        onRename={handleRename}
                                        onDelete={handleDelete}
                                        onEnableBlocked={handleEnableBlocked}
                                        actionButtonStyle={{
                                            position: 'absolute',
                                            top: '50%',
                                            right: dimensionValue('size-100'),
                                            transform: 'translateY(-50%)',
                                        }}
                                    />
                                </View>
                                {hasActivePipeline ? <Tag text={'Active'} /> : null}
                            </Flex>
                        </Header>
                    )}

                    {otherProjects.length > 0 && (
                        <>
                            <Divider size={'S'} marginBottom={'size-100'} marginTop={0} />

                            <Content margin={0}>
                                <ProjectsList
                                    projects={otherProjects}
                                    onRename={handleRename}
                                    onDelete={handleDelete}
                                    onEnableBlocked={handleEnableBlocked}
                                />
                            </Content>
                        </>
                    )}

                    <ButtonGroup UNSAFE_className={classes.buttonsGroup}>
                        <ManageProjects />
                    </ButtonGroup>
                </Dialog>
            </DialogTrigger>

            {projectActionMetadata !== null && (
                <EditProjectNameDialog
                    key={`edit-${projectActionMetadata.projectId}`}
                    projectId={projectActionMetadata.projectId}
                    projectName={projectActionMetadata.projectName}
                    projectNames={projectActionMetadata.projectNames}
                    isOpen={editProjectNameDialogState.isOpen}
                    onClose={handleCloseEdit}
                />
            )}

            {projectActionMetadata !== null && (
                <DeleteProjectDialog
                    projectId={projectActionMetadata.projectId}
                    projectName={projectActionMetadata.projectName}
                    isOpen={deleteProjectDialogState.isOpen}
                    onClose={deleteProjectDialogState.close}
                    onDeleted={handleDeleted}
                />
            )}

            <EnablePipelineBlockedDialog isOpen={isEnableBlockedDialogOpen} onClose={handleCloseEnableBlocked} />
        </>
    );
};
