// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { Flex, Loading, View } from '@geti/ui';

import { ActiveModel } from './active-model.component';
import { InferenceDevices } from './inference-devices.component';
import { TogglePipelineButton } from './toggle-pipeline-button.component';

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

                <Suspense fallback={<Loading />}>
                    <InferenceDevices />
                </Suspense>

                <Flex marginStart='auto' gap={'size-100'}>
                    <TogglePipelineButton />
                </Flex>
            </Flex>
        </View>
    );
};
