// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid, Item, TabList, TabPanels, Tabs, View } from '@geti/ui';
import { Outlet, useLocation } from 'react-router';

import { ReactComponent as BuildIcon } from './assets/icons/build-icon.svg';
import { ReactComponent as LiveFeedIcon } from './assets/icons/live-feed-icon.svg';
import { paths } from './router';

const Header = () => {
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
                        textValue='"Live feed page showing live inference on your pipeline'
                        key={paths.inference.index({})}
                        href={paths.inference.index({})}
                    >
                        <Flex alignItems='center' gap='size-100'>
                            <LiveFeedIcon />
                            Inference
                        </Flex>
                    </Item>
                    <Item
                        textValue='Data collection page to visualise your media items'
                        key={paths.dataset.index({})}
                        href={paths.dataset.index({})}
                    >
                        <Flex alignItems='center' gap='size-100'>
                            <BuildIcon />
                            Dataset
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
                        <Item textValue='index' key={paths.pipeline.index({})}>
                            <Outlet />
                        </Item>
                        <Item textValue='live-feed' key={paths.inference.index({})}>
                            <Outlet />
                        </Item>
                        <Item textValue='data-collection' key={paths.dataset.index({})}>
                            <Outlet />
                        </Item>
                    </TabPanels>
                </View>
            </Grid>
        </Tabs>
    );
};
