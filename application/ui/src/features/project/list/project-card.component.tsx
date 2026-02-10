// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Badge, dimensionValue, Flex, Heading, Text, View } from '@geti/ui';
import { clsx } from 'clsx';
import { NavLink } from 'react-router-dom';

import type { SchemaProjectView } from '../../../api/openapi-spec';
import { paths } from '../../../constants/paths';
import { TaskType } from '../../../constants/shared-types';
import { getProjectThumbnailUrl } from '../../../shared/media-url.utils';
import { isClassificationTask } from '../task-type-guards';
import { MenuActions } from './menu-actions/menu-actions.component';

import classes from './project-list.module.scss';

const cardPadding = 'size-200';

const MAP_PROJECT_TYPE_TO_TITLE: Record<TaskType, string> = {
    detection: 'Detection',
    classification: 'Classification',
    instance_segmentation: 'Instance segmentation',
};

type ProjectTypeBadgeProps = {
    type: string;
};

const ProjectTypeBadge = ({ type }: ProjectTypeBadgeProps) => {
    return (
        <Badge variant={'neutral'} UNSAFE_className={classes.tag}>
            <Text>{type}</Text>
        </Badge>
    );
};

const ActiveProjectBadge = () => {
    return (
        <Badge variant={'neutral'} UNSAFE_className={classes.activeTag}>
            <Text>Active</Text>
        </Badge>
    );
};

type ProjectCardProps = {
    item: SchemaProjectView;
};

export const ProjectCard = ({ item }: ProjectCardProps) => {
    const isActive = item.active_pipeline;
    const isClassification = isClassificationTask(item.task.task_type);
    const isMultiLabel = isClassification && item.task.exclusive_labels === false;

    return (
        <div style={{ position: 'relative' }}>
            <NavLink to={paths.project.dataset({ projectId: item.id })}>
                <Flex UNSAFE_className={clsx({ [classes.card]: true, [classes.activeCard]: isActive })}>
                    <Flex>
                        <img src={getProjectThumbnailUrl(item.id)} alt={'N/A'} />
                    </Flex>

                    <View width={'100%'} padding={cardPadding}>
                        <Flex alignItems={'center'} justifyContent={'space-between'}>
                            <Heading level={3} marginEnd={'size-400'}>
                                {item.name}
                            </Heading>
                        </Flex>

                        <Flex marginBottom={cardPadding} gap={'size-50'}>
                            {isActive && <ActiveProjectBadge />}
                            {isMultiLabel ? (
                                <ProjectTypeBadge type={'Multi-label classification'} />
                            ) : (
                                <ProjectTypeBadge type={MAP_PROJECT_TYPE_TO_TITLE[item.task.task_type]} />
                            )}
                        </Flex>

                        <Flex gap={'size-100'} direction={'column'}>
                            <Text>• Labels: {(item.task.labels ?? []).map((label) => label.name).join(', ')}</Text>
                        </Flex>
                    </View>
                </Flex>
            </NavLink>

            <MenuActions
                projectId={item.id}
                projectName={item.name}
                actionButtonStyle={{
                    top: dimensionValue(cardPadding),
                    right: dimensionValue(cardPadding),
                    position: 'absolute',
                }}
            />
        </div>
    );
};
