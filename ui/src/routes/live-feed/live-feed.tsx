import { useState } from 'react';

import { Button, Flex, Grid, Loading, View } from '@geti/ui';
import { Play } from '@geti/ui/icons';

import { useWebRTCConnection } from '../../components/stream/web-rtc-connection-provider';
import { Stream } from './../../components/stream/stream';
import { Aside } from './aside';
import { Toolbar } from './toolbar';

import classes from './live-feed.module.css';

export const StreamContainer = () => {
    const [size, setSize] = useState({ height: 608, width: 892 });
    const { start, status } = useWebRTCConnection();

    return (
        <>
            {status === 'idle' && (
                <div className={classes.canvasContainer}>
                    <View backgroundColor={'gray-200'} width='90%' height='90%'>
                        <Flex alignItems={'center'} justifyContent={'center'} height='100%'>
                            <Button onPress={start} UNSAFE_className={classes.playButton}>
                                <Play width='128px' height='128px' />
                            </Button>
                        </Flex>
                    </View>
                </div>
            )}

            {status === 'connecting' && (
                <div className={classes.canvasContainer}>
                    <View backgroundColor={'gray-200'} width='90%' height='90%'>
                        <Flex alignItems={'center'} justifyContent={'center'} height='100%'>
                            <Loading mode='inline' />
                        </Flex>
                    </View>
                </div>
            )}

            {status === 'connected' && (
                <div className={classes.canvasContainer}>
                    <Stream size={size} setSize={setSize} />
                </div>
            )}
        </>
    );
};

export const LiveFeed = () => {
    return (
        <Grid
            areas={['toolbar aside', 'canvas aside']}
            UNSAFE_style={{
                gridTemplateRows: 'var(--spectrum-global-dimension-size-800, 4rem) auto',
                gridTemplateColumns: 'auto min-content',
            }}
            height={'100%'}
            gap='1px'
        >
            <Aside />
            <Toolbar />

            <View gridArea={'canvas'} overflow={'hidden'} maxHeight={'100%'}>
                <StreamContainer />
            </View>
        </Grid>
    );
};
