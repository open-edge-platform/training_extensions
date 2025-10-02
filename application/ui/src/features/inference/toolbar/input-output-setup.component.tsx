// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { Button, Content, Dialog, DialogTrigger, Item, Loading, TabList, TabPanels, Tabs, Text, View } from '@geti/ui';

import { ReactComponent as Camera } from '../../../assets/icons/camera.svg';
import { SinkOptions } from '../sinks/sink-options';
import { SourceOptions } from '../sources/source-options';

export const InputOutputSetup = () => {
    return (
        <DialogTrigger type='popover'>
            <Button width={'size-2000'} variant={'secondary'}>
                <Camera fill='white' />
                <Text width={'auto'} marginStart={'size-100'}>
                    Input source
                </Text>
            </Button>
            <Dialog minWidth={'size-6000'}>
                <Content>
                    <Tabs aria-label='Dataset import tabs' height={'100%'}>
                        <TabList>
                            <Item key='sources' textValue='FoR'>
                                <Text>Input setup</Text>
                            </Item>
                            <Item key='sinks' textValue='MaR'>
                                <Text>Output setup</Text>
                            </Item>
                        </TabList>
                        <TabPanels>
                            <Item key='sources'>
                                <View marginTop={'size-200'}>
                                    <Suspense fallback={<Loading size='M' />}>
                                        <SourceOptions />
                                    </Suspense>
                                </View>
                            </Item>
                            <Item key='sinks'>
                                <View marginTop={'size-200'}>
                                    <SinkOptions />
                                </View>
                            </Item>
                        </TabPanels>
                    </Tabs>
                </Content>
            </Dialog>
        </DialogTrigger>
    );
};
