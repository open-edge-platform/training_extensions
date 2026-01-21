// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { Divider, Flex, View } from '@geti/ui';

import { ActiveModel } from './active-model.component';
import { InferenceDevices } from './inference-devices.component';
import { InputOutputSetup } from './input-output-setup.component';
import { PipelineSwitch } from './pipeline-toggle.component';
import { WebRTCConnectionStatus } from './web-rtc-connection-status.component';

export const Header = () => {
    return (
        <View
            backgroundColor='gray-100'
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

                <InferenceDevices />

                <Flex marginStart='auto'>
                    <PipelineSwitch />
                    <InputOutputSetup />
                </Flex>
            </Flex>
        </View>
    );
};
