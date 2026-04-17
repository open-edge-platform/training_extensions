// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, PhotoPlaceholder, Text } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { useNavigate } from 'react-router';

import type { SchemaProjectView } from '../../../api/openapi-spec';
import { paths } from '../../../constants/paths';
import { MenuActions } from '../../../features/project/list/menu-actions/menu-actions.component';
import { useProjects } from '../../../hooks/api/project.hook';

import classes from './project-list-item.module.scss';

interface ProjectListItemProps {
    project: SchemaProjectView;
}

export const ProjectListItem = ({ project }: ProjectListItemProps) => {
    const navigate = useNavigate();
    const currentProjectId = useProjectIdentifier();
    const { data: projects } = useProjects();

    const handleNavigateToProject = () => {
        navigate(paths.project.dataset.index({ projectId: project.id }));
    };

    const handleDeleted = () => {
        if (project.id === currentProjectId) {
            const remainingProjects = projects.filter((p) => p.id !== project.id);

            if (remainingProjects.length > 0) {
                navigate(paths.project.index({}));
            } else {
                navigate(paths.project.new({}));
            }
        }
    };

    const projectsNames = projects.filter(({ id }) => id !== project.id).map(({ name }) => name);

    return (
        <li className={classes.projectListItem} onClick={handleNavigateToProject}>
            <Flex justifyContent='space-between' alignItems='center' marginX={'size-200'}>
                <Flex alignItems={'center'} gap={'size-100'} minWidth={0}>
                    <PhotoPlaceholder
                        name={project.name}
                        indicator={project.id ?? project.name}
                        height={'size-300'}
                        width={'size-300'}
                    />
                    <Text UNSAFE_className={classes.projectName}>
                        <span title={project.name}>{project.name}</span>
                    </Text>
                </Flex>
                <MenuActions
                    projectId={project.id}
                    projectName={project.name}
                    isPipelineRunning={project.active_pipeline}
                    onDeleted={handleDeleted}
                    projectsNames={projectsNames}
                />
            </Flex>
        </li>
    );
};
