// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react';

import { Flex, Loading, toast, View } from '@geti/ui';
import { Pause, Play } from '@geti/ui/icons';

import { Stream } from './stream';
import { useWebRTCConnection } from './web-rtc-connection-provider';

import classes from './stream.module.scss';

export const StreamContainer = () => {
    const [size, setSize] = useState({ height: 608, width: 892 });
    const { start, stop, status } = useWebRTCConnection();

    const isStopped = status === 'idle' || status === 'failed';
    const isConnecting = status === 'connecting';
    const isConnected = status === 'connected';

    useEffect(() => {
        if (status === 'failed') {
            toast({ type: 'error', message: 'Failed to connect to the stream' });
        }
    }, [status]);

    return (
        <View gridArea={'canvas'} overflow={'hidden'} maxHeight={'100%'}>
            <div className={classes.canvasContainer} onClick={isConnected ? stop : isStopped ? start : undefined}>
                <View backgroundColor={'gray-200'} width='90%' height='90%'>
                    {isStopped && (
                        <Flex alignItems={'center'} justifyContent={'center'} height='100%'>
                            <Flex UNSAFE_className={classes.playPauseButton}>
                                <Play color={'#333'} width='64px' height='64px' aria-label={'Start stream'} />
                            </Flex>
                        </Flex>
                    )}

                    {isConnecting && (
                        <Flex alignItems={'center'} justifyContent={'center'} height='100%'>
                            <Loading mode='inline' />
                        </Flex>
                    )}

                    {isConnected && (
                        <View position='relative' width='100%' height='100%' UNSAFE_className={classes.streamWrapper}>
                            <Stream size={size} setSize={setSize} />

                            <Flex
                                position='absolute'
                                alignItems='center'
                                justifyContent='center'
                                width='100%'
                                height='100%'
                                UNSAFE_className={classes.pauseOverlay}
                            >
                                <Flex UNSAFE_className={classes.playPauseButton}>
                                    <Pause color={'#333'} width='64px' height='64px' aria-label={'Stop stream'} />
                                </Flex>
                            </Flex>
                        </View>
                    )}
                </View>
            </div>
        </View>
    );
};
