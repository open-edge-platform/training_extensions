// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { Flex, Grid, Item, Loading, TabList, Tabs, View } from '@geti/ui';
import { useProject } from 'hooks/api/project.hook';
import { Outlet, useLocation } from 'react-router';
import { Link } from 'react-router-dom';

import { ReactComponent as BuildIcon } from './assets/icons/build-icon.svg';
import { ReactComponent as LiveFeedIcon } from './assets/icons/live-feed-icon.svg';
import { ReactComponent as Webhook } from './assets/icons/webhook.svg';
import { ProjectsListPanel } from './components/project-panel/projects-list-panel.component';
import { paths } from './constants/paths';
import { useProjectIdentifier } from './hooks/use-project-identifier.hook';

import classes from './layout.module.scss';

const iconStyles = {
    width: 'var(--spectrum-global-dimension-size-200)',
    height: 'var(--spectrum-global-dimension-size-200)',
};

const Header = () => {
    const projectId = useProjectIdentifier();

    return (
        <View backgroundColor={'gray-200'} gridArea={'header'}>
            <Grid
                height='100%'
                gap={'size-200'}
                marginStart={'size-200'}
                columns={['auto', '2fr', 'size-2400']}
                rows={'1fr'}
                alignItems={'center'}
            >
                <View paddingEnd={'size-200'}>
                    <Link to={paths.project.index({})}>Geti</Link>
                </View>

                <TabList height={'100%'} UNSAFE_className={classes.tabList}>
                    <Item
                        textValue='Data collection page to visualise your media items'
                        key={'dataset'}
                        href={paths.project.dataset.index({ projectId })}
                    >
                        <Flex alignItems='center' gap='size-100'>
                            <BuildIcon style={iconStyles} />
                            Dataset
                        </Flex>
                    </Item>
                    <Item
                        textValue='Models page to visualise your models'
                        key={'models'}
                        href={paths.project.models({ projectId })}
                    >
                        <Flex alignItems='center' gap='size-100'>
                            <Webhook style={iconStyles} />
                            Models
                        </Flex>
                    </Item>
                    <Item
                        textValue='Inference page showing live inference on your project'
                        key={'inference'}
                        href={paths.project.inference({ projectId })}
                    >
                        <Flex alignItems='center' gap='size-100'>
                            <LiveFeedIcon style={iconStyles} />
                            Inference
                        </Flex>
                    </Item>
                </TabList>

                <Suspense fallback={<Loading />}>
                    <ProjectsListPanel />
                </Suspense>
            </Grid>
        </View>
    );
};

const getFirstPathSegment = (path: string): string => {
    return path.split('/').pop() || '';
};

export const Layout = () => {
    const { pathname } = useLocation();
    // We want to check if the project exists before rendering the layout. If it doesn't, error boundary will catch it.
    useProject();

    return (
        <Tabs aria-label='Header navigation' selectedKey={getFirstPathSegment(pathname)}>
            <Grid
                areas={['header', 'content']}
                rows={['size-800', 'minmax(0, 1fr)']}
                minHeight={'100vh'}
                maxHeight={'100vh'}
                height={'100%'}
            >
                <Header />

                <View backgroundColor={'gray-50'} gridArea={'content'} position={'relative'}>
                    <Suspense fallback={<Loading />}>
                        <Outlet />
                    </Suspense>
                </View>
            </Grid>
        </Tabs>
    );
};
