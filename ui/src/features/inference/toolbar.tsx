// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { StatusLight } from '@adobe/react-spectrum';
import { Button, Divider, Flex, Text, View } from '@geti/ui';

import { $api } from '../../api/client';
import { paths } from '../../router';
import { useWebRTCConnection } from './stream/web-rtc-connection-provider';

const ActiveModel = () => {
    const modelsQuery = $api.useSuspenseQuery('get', '/api/models');

    return (
        <Flex gap='size-50' alignItems='center'>
            <Text
                UNSAFE_style={{
                    color: 'var(--spectrum-global-color-gray-900)',
                }}
            >
                Model:
            </Text>
            <Text
                UNSAFE_style={{
                    color: 'var(--spectrum-global-color-gray-700)',
                }}
            >
                {modelsQuery.data.length > 0 ? modelsQuery.data[0].name : 'Unknown'}
            </Text>
        </Flex>
    );
};

const WebRTCConnectionStatus = () => {
    const { status, stop } = useWebRTCConnection();

    switch (status) {
        case 'idle':
            return (
                <Flex
                    gap='size-100'
                    alignItems={'center'}
                    UNSAFE_style={{
                        '--spectrum-gray-visual-color': 'var(--spectrum-global-color-gray-500)',
                    }}
                >
                    <StatusLight role={'status'} aria-label='Idle' variant='neutral'>
                        Idle
                    </StatusLight>
                </Flex>
            );
        case 'connecting':
            return (
                <Flex gap='size-100' alignItems={'center'}>
                    <StatusLight role={'status'} aria-label='Connecting' variant='info'>
                        Connecting
                    </StatusLight>
                </Flex>
            );
        case 'disconnected':
            return (
                <Flex gap='size-100' alignItems={'center'}>
                    <StatusLight role={'status'} aria-label='Disconnected' variant='negative'>
                        Disconnected
                    </StatusLight>
                </Flex>
            );
        case 'failed':
            return (
                <Flex gap='size-100' alignItems={'center'}>
                    <StatusLight role={'status'} aria-label='Failed' variant='negative'>
                        Failed
                    </StatusLight>
                </Flex>
            );
        case 'connected':
            return (
                <Flex gap='size-200' alignItems={'center'}>
                    <StatusLight role={'status'} aria-label='Connected' variant='positive'>
                        Connected
                    </StatusLight>
                    <Button onPress={stop} variant='secondary'>
                        Stop
                    </Button>
                </Flex>
            );
    }
};

export const Toolbar = () => {
    return (
        <View
            backgroundColor={'gray-100'}
            gridArea='toolbar'
            padding='size-200'
            UNSAFE_style={{
                fontSize: '12px',
                color: 'var(--spectrum-global-color-gray-800)',
            }}
        >
            <Flex height='100%' gap='size-200' alignItems={'center'}>
                <Suspense fallback={'Model: ...'}>
                    <ActiveModel />
                </Suspense>

                <Divider orientation='vertical' size='S' />

                <WebRTCConnectionStatus />

                <Divider orientation='vertical' size='S' />

                <Flex marginStart='auto' gap='size-100'>
                    <Button href={paths.project.index({})} variant='secondary'>
                        View project
                    </Button>
                </Flex>
            </Flex>
        </View>
    );
};
