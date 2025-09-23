// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid, Item, TabList, TabPanels, Tabs, View } from '@geti/ui';
import { Outlet, useLocation } from 'react-router';
import { Link } from 'react-router-dom';

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
                    <Link to={paths.project.index({})}>Geti Tune</Link>
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
                        key={'inference'}
                        href={paths.project.inference({ projectId })}
                    >
                        <Flex alignItems='center' gap='size-100'>
                            <LiveFeedIcon style={iconStyles} />
                            Inference
                        </Flex>
                    </Item>
                    <Item
                        textValue='Data collection page to visualise your media items'
                        key={'dataset'}
                        href={paths.project.dataset({ projectId })}
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
                </TabList>
            </Flex>
        </View>
    );
};

const getFirstPathSegment = (path: string): string => {
    return path.split('/').pop() || '';
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
                        <Item textValue='index' key={'index'}>
                            <Outlet />
                        </Item>
                        <Item textValue='inference' key={'inference'}>
                            <Outlet />
                        </Item>
                        <Item textValue='dataset' key={'dataset'}>
                            <Outlet />
                        </Item>
                        <Item textValue='models' key={'models'}>
                            <Outlet />
                        </Item>
                    </TabPanels>
                </View>
            </Grid>
        </Tabs>
    );
};
