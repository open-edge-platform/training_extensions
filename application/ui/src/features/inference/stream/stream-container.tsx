// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { dimensionValue, Flex, Loading, Text, toast, View } from '@geti/ui';
import { Pause, Play } from '@geti/ui/icons';
import { clsx } from 'clsx';

import { usePipeline } from '../../../hooks/api/pipeline.hook';
import { Stream } from './stream';
import { useWebRTCConnection } from './web-rtc-connection-provider';

import classes from './stream.module.scss';

export const StreamContainer = () => {
    const { start, stop, status, webRTCConnectionRef } = useWebRTCConnection();
    const { data: pipeline } = usePipeline();

    const isPipelineRunning = pipeline.status === 'running';
    const isStopped = status === 'idle' || status === 'failed';
    const isConnecting = status === 'connecting';
    const isConnected = status === 'connected';

    const [isPaused, setIsPaused] = useState(false);

    const canStart = isStopped && isPipelineRunning;

    const handleClick = async () => {
        if (isConnected) {
            setIsPaused(true);
            await stop();
        } else if (canStart) {
            setIsPaused(false);
            await start();

            if (webRTCConnectionRef.current?.getStatus() === 'failed') {
                toast({ type: 'error', message: 'Failed to connect to the stream' });
            }
        }
    };

    return (
        <View gridArea={'canvas'} overflow={'hidden'} maxHeight={'100%'}>
            <div
                className={classes.canvasContainer}
                onClick={handleClick}
                title={isPipelineRunning ? undefined : 'Enable pipeline to start stream'}
            >
                {isStopped && (
                    <Flex justifyContent={'center'} alignItems={'center'} UNSAFE_className={classes.backdrop}>
                        <Flex
                            justifyContent={'center'}
                            alignItems={'center'}
                            UNSAFE_className={clsx(classes.playPauseButtonWrapper, {
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
                            <Text UNSAFE_style={{ paddingRight: dimensionValue('size-100') }}>Start stream</Text>
                        </Flex>
                    </Flex>
                )}

                {isConnecting && (
                    <Flex alignItems={'center'} justifyContent={'center'} height='100%'>
                        <Loading mode='inline' />
                    </Flex>
                )}

                {isConnected && (
                    <>
                        {!isPaused && <Stream />}
                        <Flex
                            alignItems='center'
                            justifyContent='center'
                            UNSAFE_className={clsx(classes.pauseFlash, { [classes.pauseFlashActive]: isPaused })}
                        >
                            <Flex UNSAFE_className={clsx(classes.playPauseButtonWrapper, classes.pauseFlashButton)}>
                                <Pause
                                    color={'currentColor'}
                                    width={dimensionValue('size-400')}
                                    height={dimensionValue('size-400')}
                                    aria-label={'Stream stopped'}
                                />
                            </Flex>
                        </Flex>
                    </>
                )}
            </div>
        </View>
    );
};
