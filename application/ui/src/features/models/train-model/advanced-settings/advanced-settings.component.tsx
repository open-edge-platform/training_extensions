// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Item, TabList, TabPanels, Tabs, Text, View } from '@geti/ui';

import { useTrainModel } from '../train-model-provider.component';
import { DataManagement } from './data-management/data-management.component';
import { Training } from './training/training.component';

type ContentWrapperProps = { children: ReactNode };

const ContentWrapper = ({ children }: ContentWrapperProps) => {
    return (
        <View padding={'size-250'} backgroundColor={'gray-50'} overflow={'hidden auto'} height={'100%'}>
            {children}
        </View>
    );
};

type TabProps = {
    name: string;
    children: ReactNode;
};

export const AdvancedSettings = () => {
    const { trainingConfiguration, onTrainingConfigurationChange } = useTrainModel();

    const TABS: TabProps[] = [
        {
            name: 'Data management',
            children: <DataManagement />,
        },
        {
            name: 'Training',
            children: <Training />,
        },
    ];

    return (
        <Tabs items={TABS} height={'100%'} UNSAFE_style={{ overflow: 'hidden' }} aria-label={'Advanced settings tabs'}>
            <TabList>
                {(tab: TabProps) => (
                    <Item key={tab.name} textValue={tab.name}>
                        <Text>{tab.name}</Text>
                    </Item>
                )}
            </TabList>
            <TabPanels marginTop={'size-250'} UNSAFE_style={{ overflow: 'hidden' }}>
                {(tab: TabProps) => (
                    <Item key={tab.name} textValue={tab.name}>
                        <ContentWrapper>{tab.children}</ContentWrapper>
                    </Item>
                )}
            </TabPanels>
        </Tabs>
    );
};
