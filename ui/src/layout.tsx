// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid, Item, TabList, TabPanels, Tabs, View } from '@geti/ui';
import { Outlet, useLocation } from 'react-router';

import { ReactComponent as BuildIcon } from './assets/icons/build-icon.svg';
import { ReactComponent as LiveFeedIcon } from './assets/icons/live-feed-icon.svg';
import { ReactComponent as Webhook } from './assets/icons/webhook.svg';
import { useProjectIdentifier } from './hooks/use-project-identifier.hook';
import { paths } from './router';

const iconStyles = {
    width: 'var(--spectrum-global-dimension-size-200)',
    height: 'var(--spectrum-global-dimension-size-200)',
};

const Header = () => {
    const projectId = useProjectIdentifier();

    return (
        <View backgroundColor={'gray-300'} gridArea={'header'}>
            <Flex height='100%' alignItems={'center'} marginX='1rem' gap='size-200'>
                <View marginEnd='size-200'>
                    <span>Geti Tune</span>
                </View>

                <TabList
                    height={'100%'}
                    UNSAFE_style={{
                        '--spectrum-tabs-rule-height': '4px',
                        '--spectrum-tabs-selection-indicator-color': 'var(--energy-blue)',
                    }}
                >
                    <Item
                        textValue='Inference page showing live inference on your project'
                        key={paths.project.inference({ projectId })}
                        href={paths.project.inference({ projectId })}
                    >
                        <Flex alignItems='center' gap='size-100'>
                            <LiveFeedIcon style={iconStyles} />
                            Inference
                        </Flex>
                    </Item>
                    <Item
                        textValue='Data collection page to visualise your media items'
                        key={paths.project.dataset({ projectId })}
                        href={paths.project.dataset({ projectId })}
                    >
                        <Flex alignItems='center' gap='size-100'>
                            <BuildIcon style={iconStyles} />
                            Dataset
                        </Flex>
                    </Item>
                    <Item
                        textValue='Models page to visualise your models'
                        key={paths.project.models({ projectId })}
                        href={paths.project.models({ projectId })}
                    >
                        <Flex alignItems='center' gap='size-100'>
                            <Webhook style={iconStyles} />
                            Models
                        </Flex>
                    </Item>
                </TabList>
            </Flex>
        </View>
    );
};

const getFirstPathSegment = (path: string): string => {
    const segments = path.split('/');

    return segments.length > 1 ? `/${segments[1]}` : '/';
};

export const Layout = () => {
    const { pathname } = useLocation();
    const projectId = useProjectIdentifier();

    return (
        <Tabs aria-label='Header navigation' selectedKey={getFirstPathSegment(pathname)}>
            <Grid
                areas={['header', 'content']}
                UNSAFE_style={{
                    gridTemplateRows: 'var(--spectrum-global-dimension-size-800, 4rem) auto',
                }}
                minHeight={'100vh'}
                maxHeight={'100vh'}
                height={'100%'}
            >
                <Header />
                <View backgroundColor={'gray-50'} gridArea={'content'}>
                    <TabPanels height={'100%'} UNSAFE_style={{ border: 'none' }}>
                        <Item textValue='index' key={paths.project.index({})}>
                            <Outlet />
                        </Item>
                        <Item textValue='inference' key={paths.project.inference({ projectId })}>
                            <Outlet />
                        </Item>
                        <Item textValue='dataset' key={paths.project.dataset({ projectId })}>
                            <Outlet />
                        </Item>
                        <Item textValue='models' key={paths.project.models({ projectId })}>
                            <Outlet />
                        </Item>
                    </TabPanels>
                </View>
            </Grid>
        </Tabs>
    );
};
