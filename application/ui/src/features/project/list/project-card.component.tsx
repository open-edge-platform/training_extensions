// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Heading, Tag, Text, View } from '@geti/ui';
import { clsx } from 'clsx';
import { NavLink } from 'react-router-dom';

import { SchemaProjectView } from '../../../api/openapi-spec';
import thumbnailUrl from '../../../assets/mocked-project-thumbnail.png';
import { paths } from '../../../constants/paths';
import { MenuActions } from './menu-actions.component';

import classes from './project-list.module.scss';

type ProjectCardProps = {
    item: SchemaProjectView;
};

export const ProjectCard = ({ item }: ProjectCardProps) => {
    const isActive = item.active_pipeline;

    return (
        <NavLink to={paths.project.inference({ projectId: item.id })}>
            <Flex UNSAFE_className={clsx({ [classes.card]: true, [classes.activeCard]: isActive })}>
                <View aria-label={'project thumbnail'}>
                    <img src={thumbnailUrl} alt={item.name} />
                </View>

                <View width={'100%'} padding={'size-200'}>
                    <Flex alignItems={'center'} justifyContent={'space-between'}>
                        <Heading level={3}>{item.name}</Heading>
                        <MenuActions projectId={item.id} />
                    </Flex>

                    <Flex marginBottom={'size-200'} gap={'size-50'}>
                        {isActive && (
                            <Tag withDot={false} text='Active' className={clsx(classes.tag, classes.activeTag)} />
                        )}
                        <Tag withDot={false} text={item.task.task_type} className={classes.tag} />
                    </Flex>

                    <Flex alignItems={'center'} gap={'size-100'} direction={'row'} wrap='wrap'>
                        <Text>• Edited: 2025-08-07 06:05 AM</Text>
                        <Text>• Labels: {(item.task.labels || []).map((label) => label.name).join(', ')}</Text>
                    </Flex>
                </View>
            </Flex>
        </NavLink>
    );
};
