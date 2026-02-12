// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, PhotoPlaceholder, Text } from '@geti/ui';
import { useNavigate } from 'react-router';

import type { SchemaProjectView } from '../../../api/openapi-spec';
import { paths } from '../../../constants/paths';
import { MenuActions } from '../../../features/project/list/menu-actions/menu-actions.component';

import classes from './project-list-item.module.scss';

interface ProjectListItemProps {
    project: SchemaProjectView;
}

export const ProjectListItem = ({ project }: ProjectListItemProps) => {
    const navigate = useNavigate();

    const handleNavigateToProject = () => {
        navigate(paths.project.dataset({ projectId: project.id }));
    };

    return (
        <li className={classes.projectListItem} onClick={handleNavigateToProject}>
            <Flex justifyContent='space-between' alignItems='center' marginX={'size-200'}>
                <Flex alignItems={'center'} gap={'size-100'}>
                    <PhotoPlaceholder
                        name={project.name}
                        indicator={project.id ?? project.name}
                        height={'size-300'}
                        width={'size-300'}
                    />
                    <Text>{project.name}</Text>
                </Flex>
                <MenuActions projectId={project.id} projectName={project.name} />
            </Flex>
        </li>
    );
};
