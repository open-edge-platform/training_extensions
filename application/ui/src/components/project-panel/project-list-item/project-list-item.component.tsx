// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Badge, Flex, Text } from '@geti/ui';
import { useNavigate } from 'react-router';

import { paths } from '../../../constants/paths';
import { Project } from '../../../constants/shared-types';
import { MenuActions } from '../../../features/project/list/menu-actions/menu-actions.component';
import { MAP_PROJECT_TYPE_TO_TITLE } from '../../../features/project/list/util';
import { ProjectThumbnail } from '../project-thumbnail/project-thumbnail.component';

import classes from './project-list-item.module.scss';

interface ProjectListItemProps {
    project: Project;
    projectNames: string[];
}

export const ProjectListItem = ({ project, projectNames }: ProjectListItemProps) => {
    const navigate = useNavigate();

    const taskType = MAP_PROJECT_TYPE_TO_TITLE[project.task.task_type];

    const handleNavigateToProject = () => {
        navigate(paths.project.dataset.index({ projectId: project.id }));
    };

    return (
        <li className={classes.projectListItem} onClick={handleNavigateToProject}>
            <Flex justifyContent='space-between' alignItems='center' marginX={'size-200'}>
                <Flex alignItems={'center'} gap={'size-100'} minWidth={0}>
                    <ProjectThumbnail project={project} height={'size-300'} width={'size-300'} />
                    <Text UNSAFE_className={classes.projectName}>
                        <span title={project.name}>{project.name}</span>
                    </Text>
                    <Badge variant={'neutral'} UNSAFE_className={classes.itemTag}>
                        <Text>{taskType}</Text>
                    </Badge>
                </Flex>

                <MenuActions
                    projectId={project.id}
                    projectName={project.name}
                    isPipelineRunning={project.active_pipeline}
                    projectNames={projectNames}
                />
            </Flex>
        </li>
    );
};
