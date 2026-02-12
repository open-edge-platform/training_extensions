// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { Content, Grid, Heading, Loading, Text, View } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { useProjects } from '../../../hooks/api/project.hook';
import { NewProjectLink } from './new-project-link.component';
import { ProjectCard } from './project-card.component';

import backgroundStyles from '../project-background.module.scss';
import classes from './project-list.module.scss';

const ProjectGrid = () => {
    const projects = useProjects();

    return (
        <Grid
            gap={'size-300'}
            height={'100%'}
            marginX={'auto'}
            maxHeight={'75vh'}
            autoRows={'size-2000'}
            justifyContent={'center'}
            UNSAFE_style={{ overflow: 'auto' }}
            columns={isEmpty(projects.data) ? ['size-3600'] : ['1fr', '1fr']}
        >
            <NewProjectLink />
            {projects.data.map((item) => (
                <ProjectCard key={item.id} item={item} />
            ))}
        </Grid>
    );
};

export const ProjectList = () => {
    return (
        <View UNSAFE_className={backgroundStyles.projectBackground} paddingTop={'size-1000'} height={'100%'}>
            <Content height={'100%'} maxHeight={'90vh'} maxWidth={'1052px'} margin={'0 auto'}>
                <Heading
                    level={1}
                    marginBottom={'size-250'}
                    UNSAFE_style={{
                        textAlign: 'center',
                        fontSize: 'var(--spectrum-global-dimension-font-size-700)',
                    }}
                >
                    Projects
                </Heading>

                <Text UNSAFE_className={classes.description}>
                    To create a project, start by defining your objectives. Then, design the data flow to ensure proper
                    processing at each stage. Implement the required tools and technologies for automation, and finally,
                    test the project to confirm it runs smoothly and meets your goals.
                </Text>

                <Suspense fallback={<Loading size='M' mode='inline' />}>
                    <ProjectGrid />
                </Suspense>
            </Content>
        </View>
    );
};
