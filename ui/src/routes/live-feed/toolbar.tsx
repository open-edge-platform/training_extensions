import { Suspense } from 'react';

import { StatusLight } from '@adobe/react-spectrum';
import { Button, Divider, Flex, Text, View } from '@geti/ui';

import { $api } from '../../api/client';
import { useWebRTCConnection } from '../../components/stream/web-rtc-connection-provider';
import { DebugTrigger } from './debug-trigger';

function ActiveModel() {
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
                {modelsQuery.data.active_model}
            </Text>
        </Flex>
    );
}

function WebRTCConnectionStatus() {
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
                    <StatusLight variant='neutral'>Idle</StatusLight>
                </Flex>
            );
        case 'connecting':
            return (
                <Flex gap='size-100' alignItems={'center'}>
                    <StatusLight variant='info'>Connecting</StatusLight>
                </Flex>
            );
        case 'disconnected':
            return (
                <Flex gap='size-100' alignItems={'center'}>
                    <StatusLight variant='negative'>Disconnected</StatusLight>
                </Flex>
            );
        case 'failed':
            return (
                <Flex gap='size-100' alignItems={'center'}>
                    <StatusLight variant='negative'>Failed</StatusLight>
                </Flex>
            );
        case 'connected':
            return (
                <Flex gap='size-200' alignItems={'center'}>
                    <StatusLight variant='positive'>Connected</StatusLight>
                    <Button onPress={stop} variant='secondary'>
                        Stop
                    </Button>
                </Flex>
            );
    }
}

export function Toolbar() {
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

                <DebugTrigger />
            </Flex>
        </View>
    );
}
