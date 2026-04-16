// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, Suspense } from 'react';

import { Item, Loading, TabList, TabPanels, Tabs, Text, View } from '@geti/ui';

import { SinkActions } from '../sinks/sink-actions.component';
import { SourceActions } from '../sources/source-actions.component';

const ConfigurationItem = ({ children }: { children: ReactNode }) => {
    return (
        <View position={'relative'} height={'100%'}>
            <Suspense
                fallback={<Loading style={{ backgroundColor: 'transparent' }} marginTop={'size-200'} size={'L'} />}
            >
                {children}
            </Suspense>
        </View>
    );
};

export const PipelineConfiguration = () => {
    return (
        <Tabs aria-label={'Pipeline configuration tabs'} height={'100%'}>
            <TabList marginBottom={'size-200'}>
                <Item key='sources' textValue='Sources'>
                    <Text>Input</Text>
                </Item>
                <Item key='sinks' textValue='Sinks'>
                    <Text>Output</Text>
                </Item>
            </TabList>
            <TabPanels>
                <Item key='sources'>
                    <ConfigurationItem>
                        <SourceActions />
                    </ConfigurationItem>
                </Item>
                <Item key='sinks'>
                    <ConfigurationItem>
                        <SinkActions />
                    </ConfigurationItem>
                </Item>
            </TabPanels>
        </Tabs>
    );
};
