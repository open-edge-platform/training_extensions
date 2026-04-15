// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { dimensionValue, Flex, Loading, toast, View } from '@geti/ui';
import { Pause, Play } from '@geti/ui/icons';
import { clsx } from 'clsx';

import { usePipeline } from '../../../hooks/api/pipeline.hook';
import { Stream } from './stream';
import { useWebRTCConnection } from './web-rtc-connection-provider';

import classes from './stream.module.scss';

export const StreamContainer = () => {
    const { start, stop, status } = useWebRTCConnection();
    const { data: pipeline } = usePipeline();

    const isPipelineRunning = pipeline?.status === 'running';
    const isStopped = status === 'idle' || status === 'failed';
    const isConnecting = status === 'connecting';
    const isConnected = status === 'connected';

    const canStart = isStopped && isPipelineRunning;
    const handleClick = isConnected ? stop : canStart ? start : undefined;

    useEffect(() => {
        if (status === 'failed') {
            toast({ type: 'error', message: 'Failed to connect to the stream' });
        }
    }, [status]);

    return (
        <View gridArea={'canvas'} overflow={'hidden'} maxHeight={'100%'}>
            <div className={classes.canvasContainer} onClick={handleClick}>
                <View backgroundColor={'gray-200'} height={'100%'}>
                    {isStopped && (
                        <Flex alignItems={'center'} justifyContent={'center'} height='100%'>
                            <Flex
                                UNSAFE_className={clsx(classes.playPauseButton, {
                                    [classes.playButtonDisabled]: !isPipelineRunning,
                                })}
                            >
                                <Play
                                    color={'currentColor'}
                                    width={dimensionValue('size-400')}
                                    height={dimensionValue('size-400')}
                                    aria-label={isPipelineRunning ? 'Start stream' : 'Enable pipeline to start stream'}
                                    aria-disabled={!isPipelineRunning}
                                />
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
                            <Stream />

                            <Flex
                                position='absolute'
                                alignItems='center'
                                justifyContent='center'
                                width='100%'
                                height='100%'
                                UNSAFE_className={classes.overlay}
                            >
                                <Flex UNSAFE_className={classes.playPauseButton}>
                                    <Pause
                                        color={'currentColor'}
                                        width={dimensionValue('size-400')}
                                        height={dimensionValue('size-400')}
                                        aria-label={'Stop stream'}
                                    />
                                </Flex>
                            </Flex>
                        </View>
                    )}
                </View>
            </div>
        </View>
    );
};
