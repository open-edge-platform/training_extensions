import { useState } from 'react';

import { Flex, Grid, Item, TabList, TabPanels, Tabs, View } from '@geti/ui';

import { ReactComponent as BuildIcon } from './assets/build-icon.svg';
import { ReactComponent as LiveFeedIcon } from './assets/life-feed-icon.svg';

import './index.css';

import { BuildPipelinePage } from './routes/build-pipeline/build-pipeline-page';
import { LiveFeedPage } from './routes/live-feed/live-feed-page';

function Header() {
    return (
        <View backgroundColor={'gray-300'} gridArea={'header'}>
            <Flex height='100%' alignItems={'center'} marginX='1rem' gap='size-100'>
                <span>Geti Edge</span>

                <TabList
                    width='100%'
                    height='100%'
                    UNSAFE_style={{
                        '--spectrum-tabs-rule-height': '4px',
                        '--spectrum-tabs-selection-indicator-color': 'var(--energy-blue)',
                    }}
                >
                    <Item key='build'>
                        <Flex alignItems='center' gap='size-100'>
                            <BuildIcon />
                            Build
                        </Flex>
                    </Item>
                    <Item key='live-feed'>
                        <Flex alignItems='center' gap='size-100'>
                            <LiveFeedIcon />
                            Livefeed
                        </Flex>
                    </Item>
                </TabList>
            </Flex>
        </View>
    );
}

export function App() {
    const [page, setPage] = useState<'build' | 'live-feed'>('build');
    const submitPipeline = () => {
        setPage('live-feed');
    };

    return (
        <Tabs
            aria-label='Aside view'
            onSelectionChange={(selection) => (selection === 'build' ? setPage('build') : setPage('live-feed'))}
            selectedKey={page}
        >
            <Grid
                areas={['header', 'content']}
                UNSAFE_style={{
                    gridTemplateRows: 'var(--spectrum-global-dimension-size-800, 4rem) auto',
                }}
                minHeight={'100vh'}
                height={'100%'}
            >
                <Header />
                <View backgroundColor={'gray-50'} gridArea={'content'}>
                    <TabPanels height={'100%'}>
                        <Item key='build'>
                            <BuildPipelinePage submitPipeline={submitPipeline} />
                        </Item>
                        <Item key='live-feed'>
                            <LiveFeedPage />
                        </Item>
                    </TabPanels>
                </View>
            </Grid>
        </Tabs>
    );
}
