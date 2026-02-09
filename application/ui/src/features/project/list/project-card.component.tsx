// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Flex, Heading, Tag, Text, View } from '@geti/ui';
import { clsx } from 'clsx';
import { NavLink } from 'react-router-dom';

import type { SchemaProjectView } from '../../../api/openapi-spec';
// TODO: replace mock thumbnail once /api/projects/{project_id}/thumbnail is finished
import thumbnailUrl from '../../../assets/mocked-project-thumbnail.png';
import { paths } from '../../../constants/paths';
import { MenuActions } from './menu-actions/menu-actions.component';

import classes from './project-list.module.scss';

type ProjectCardProps = {
    item: SchemaProjectView;
};

const cardPadding = 'size-200';

export const ProjectCard = ({ item }: ProjectCardProps) => {
    const isActive = item.active_pipeline;

    return (
        <div style={{ position: 'relative' }}>
            <NavLink to={paths.project.dataset({ projectId: item.id })}>
                <Flex UNSAFE_className={clsx({ [classes.card]: true, [classes.activeCard]: isActive })}>
                    <View aria-label={'project thumbnail'}>
                        <img src={thumbnailUrl} alt={item.name} />
                    </View>

                    <View width={'100%'} padding={cardPadding}>
                        <Flex alignItems={'center'} justifyContent={'space-between'}>
                            <Heading level={3} marginEnd={'size-400'}>
                                {item.name}
                            </Heading>
                        </Flex>

                        <Flex marginBottom={cardPadding} gap={'size-50'}>
                            {isActive && (
                                <Tag withDot={false} text='Active' className={clsx(classes.tag, classes.activeTag)} />
                            )}
                            <Tag withDot={false} text={item.task.task_type} className={classes.tag} />
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
