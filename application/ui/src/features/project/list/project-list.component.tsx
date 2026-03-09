// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { Content, Flex, Grid, Heading, Loading, Text, View } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { useProjects } from '../../../hooks/api/project.hook';
import { ImportDatasetDialogProvider } from '../providers/import-dataset-dialog-provider.component';
import { NewProjectMenu } from './new-project-menu.component';
import { ProjectCard } from './project-card.component';

import backgroundStyles from '../project-background.module.scss';
import classes from './project-list.module.scss';

const ProjectGrid = () => {
    const projects = useProjects();

    return (
        <Grid
            gap={'size-300'}
            marginX={'auto'}
            autoRows={'size-2000'}
            height={'100%'}
            justifyContent={'center'}
            UNSAFE_style={{ overflowY: 'auto' }}
            columns={isEmpty(projects.data) ? ['size-3600'] : ['1fr', '1fr']}
        >
            <NewProjectMenu />

            {projects.data.map((item) => (
                <ProjectCard key={item.id} item={item} />
            ))}
        </Grid>
    );
};

export const ProjectList = () => {
    return (
        <View UNSAFE_className={backgroundStyles.projectBackground} height={'100%'}>
            <Content height={'100%'} maxWidth={'1052px'} margin={'0 auto'} UNSAFE_className={classes.content}>
                <Flex direction={'column'} height={'100%'}>
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
                        Create projects to configure new computer vision pipelines. <br />
                        You can switch between the projects at any time to manage the configured pipelines.
                    </Text>

                    <View flex={1} UNSAFE_style={{ overflow: 'auto' }}>
                        <Suspense fallback={<Loading size='M' mode='inline' />}>
                            <ImportDatasetDialogProvider>
                                <ProjectGrid />
                            </ImportDatasetDialogProvider>
                        </Suspense>
                    </View>
                </Flex>
            </Content>
        </View>
    );
};
