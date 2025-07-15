import { Flex, Grid, Item, TabList, TabPanels, Tabs, View } from '@geti/ui';
import { Outlet, useLocation } from 'react-router';

import { ReactComponent as BuildIcon } from './assets/build-icon.svg';
import { ReactComponent as LiveFeedIcon } from './assets/live-feed-icon.svg';
import { paths } from './router';

const Header = () => {
    return (
        <View backgroundColor={'gray-300'} gridArea={'header'}>
            <Flex height='100%' alignItems={'center'} marginX='1rem' gap='size-200'>
                <View marginEnd='size-200'>
                    <span>Geti Edge</span>
                </View>

                <TabList
                    height={'100%'}
                    UNSAFE_style={{
                        '--spectrum-tabs-rule-height': '4px',
                        '--spectrum-tabs-selection-indicator-color': 'var(--energy-blue)',
                    }}
                >
                    <Item key={paths.liveFeed.index({})} href={paths.liveFeed.index({})}>
                        <Flex alignItems='center' gap='size-100'>
                            <LiveFeedIcon />
                            Livefeed
                        </Flex>
                    </Item>
                    <Item key={paths.dataCollection.index({})} href={paths.dataCollection.index({})}>
                        <Flex alignItems='center' gap='size-100'>
                            <BuildIcon />
                            Data collection
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
                    overflowY: 'auto',
                }}
                minHeight={'100vh'}
                maxHeight={'100vh'}
                height={'100%'}
            >
                <Header />
                <View backgroundColor={'gray-50'} gridArea={'content'}>
                    <TabPanels height={'100%'} UNSAFE_style={{ border: 'none' }}>
                        <Item key={paths.pipeline.index({})}>
                            <Outlet />
                        </Item>
                        <Item key={paths.liveFeed.index({})}>
                            <Outlet />
                        </Item>
                        <Item key={paths.dataCollection.index({})}>
                            <Outlet />
                        </Item>
                    </TabPanels>
                </View>
            </Grid>
        </Tabs>
    );
};
