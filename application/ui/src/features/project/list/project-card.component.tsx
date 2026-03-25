// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Badge, dimensionValue, Flex, Heading, Text, View } from '@geti/ui';
import { clsx } from 'clsx';
import { NavLink } from 'react-router-dom';

import { paths } from '../../../constants/paths';
import { Project, TaskType } from '../../../constants/shared-types';
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
    item: Project;
    prioritizeImage?: boolean;
};

const DEFAULT_THUMBNAIL_SIZE = 156;

export const ProjectCard = ({ item, prioritizeImage = false }: ProjectCardProps) => {
    const isActive = item.active_pipeline;
    const isClassification = isClassificationTask(item.task.task_type);
    const isMultiLabel = isClassification && item.task.exclusive_labels === false;

    return (
        <div style={{ position: 'relative' }}>
            <NavLink to={paths.project.dataset.index({ projectId: item.id })}>
                <Flex UNSAFE_className={clsx({ [classes.card]: true, [classes.activeCard]: isActive })}>
                    <Flex>
                        <img
                            src={getProjectThumbnailUrl(item.id)}
                            alt={item.name}
                            width={DEFAULT_THUMBNAIL_SIZE}
                            height={DEFAULT_THUMBNAIL_SIZE}
                            loading={prioritizeImage ? 'eager' : 'lazy'}
                            fetchPriority={prioritizeImage ? 'high' : 'auto'}
                        />
                    </Flex>

                    <View width={'100%'} padding={cardPadding}>
                        <Flex alignItems={'center'} justifyContent={'space-between'}>
                            <Heading level={2} marginEnd={'size-400'} UNSAFE_className={classes.projectName}>
                                <span title={item.name}>{item.name}</span>
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
                            <Text UNSAFE_className={classes.labelList}>
                                • Labels: {(item.task.labels ?? []).map((label) => label.name).join(', ')}
                            </Text>
                        </Flex>
                    </View>
                </Flex>
            </NavLink>

            <MenuActions
                projectId={item.id}
                projectName={item.name}
                isPipelineRunning={item.active_pipeline}
                actionButtonStyle={{
                    top: dimensionValue(cardPadding),
                    right: dimensionValue(cardPadding),
                    position: 'absolute',
                }}
            />
        </div>
    );
};
