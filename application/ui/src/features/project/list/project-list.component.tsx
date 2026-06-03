// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useMemo, useState } from 'react';

import { Content, Flex, Grid, Heading, Loading, Text, View } from '@geti/ui';
import { useProjects } from 'hooks/api/project.hook';

import { version } from '../../../../package.json';
import { isNonEmptyArray } from '../../../shared/util';
import { EmptyProjectList } from './empty-project-list/empty-project-list.component';
import { ImportJobsList } from './import-jobs-list/import-jobs-list.component';
import { NewProjectCard } from './new-project-card/new-project-card.component';
import { ProjectCard } from './project-card.component';
import { SORT_BY_HANDLERS, SortProjects } from './sort-projects/sort-projects.component';
import { SortBy } from './sort-projects/utils';

import backgroundStyles from '../project-background.module.scss';
import classes from './project-list.module.scss';

const ProjectGrid = () => {
    const projects = useProjects();
    const [sortBy, setSortBy] = useState<SortBy>('createdAt-descending');
    const hasProjects = isNonEmptyArray(projects.data);

    const sortedProjects = useMemo(() => {
        return SORT_BY_HANDLERS[sortBy](projects.data);
    }, [projects.data, sortBy]);

    const projectNames = projects.data.map((project) => project.name);

    if (!hasProjects) {
        return <EmptyProjectList />;
    }

    return (
        <Flex direction={'column'} gap={'size-100'} height={'100%'}>
            <SortProjects sortBy={sortBy} onSort={setSortBy} />

            <Grid
                flex={1}
                gap={'size-300'}
                autoRows={'size-2000'}
                justifyContent={'center'}
                UNSAFE_style={{ overflowY: 'auto' }}
                columns={['1fr', '1fr']}
            >
                <NewProjectCard />

                {sortedProjects.map((item, index) => (
                    <ProjectCard
                        key={item.id}
                        item={item}
                        prioritizeImage={index === 0}
                        projectNames={projectNames.filter((projectName) => projectName !== item.name)}
                    />
                ))}
            </Grid>
        </Flex>
    );
};

const AppInfo = () => {
    return <Text UNSAFE_className={classes.version}>v{version}</Text>;
};

export const ProjectList = () => {
    return (
        <View UNSAFE_className={backgroundStyles.projectBackground} height={'100%'} position={'relative'}>
            <Content height={'100%'} maxWidth={'1052px'} margin={'0 auto'} UNSAFE_className={classes.content}>
                <Flex direction={'column'} height={'100%'}>
                    <ImportJobsList />

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
                            <ProjectGrid />
                        </Suspense>
                    </View>
                </Flex>

                <View bottom={'size-150'} left={'size-150'} position={'absolute'}>
                    <AppInfo />
                </View>
            </Content>
        </View>
    );
};
