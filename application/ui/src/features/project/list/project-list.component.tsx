// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Content, Grid, Heading, Text, View } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { useProjects } from '../../../hooks/api/project.hook';
import { NewProjectLink } from './new-project-link.component';
import { ProjectCard } from './project-card.component';

import backgroundStyles from '../project-background.module.scss';
import classes from './project-list.module.scss';

export const ProjectList = () => {
    const projects = useProjects();

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

                <Grid
                    gap={'size-300'}
                    marginX={'auto'}
                    justifyContent={'center'}
                    columns={isEmpty(projects.data) ? ['size-3600'] : ['1fr', '1fr']}
                    UNSAFE_style={{ overflow: 'auto' }}
                    maxHeight={'75vh'}
                    height={'100%'}
                    autoRows={'size-2400'}
                >
                    <NewProjectLink />
                    {projects.data.map((item) => (
                        <ProjectCard key={item.id} item={item} />
                    ))}
                </Grid>
            </Content>
        </View>
    );
};
