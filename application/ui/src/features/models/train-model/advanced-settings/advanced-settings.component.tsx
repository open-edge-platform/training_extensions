// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, RefObject, useRef } from 'react';

import { DOMRefValue, Item, TabList, TabPanels, Tabs, Text, useUnwrapDOMRef, View } from '@geti-ui/ui';

import { useTrainModelState } from '../train-model-provider.component';
import { DataManagement } from './data-management/data-management.component';
import { Training } from './training/training.component';

type ContentWrapperProps = { children: ReactNode; ref: RefObject<DOMRefValue<HTMLDivElement> | null> };

const ContentWrapper = ({ children, ref }: ContentWrapperProps) => {
    return (
        <View ref={ref} backgroundColor={'gray-50'} overflow={'hidden auto'} height={'100%'}>
            {children}
        </View>
    );
};

type TabProps = {
    name: string;
    children: ReactNode;
};

export const AdvancedSettings = () => {
    const { trainingConfiguration, onTrainingConfigurationChange, defaultTrainingConfiguration } = useTrainModelState();
    const containerRef = useRef<DOMRefValue<HTMLDivElement>>(null);
    const unwrappedContainerRef = useUnwrapDOMRef(containerRef);

    // Should never happen, but just in case, to prevent errors in the UI
    if (trainingConfiguration === undefined || defaultTrainingConfiguration === undefined) {
        return <Text>Training configuration is not available.</Text>;
    }

    const TABS: TabProps[] = [
        {
            name: 'Data management',
            children: (
                <DataManagement
                    containerRef={unwrappedContainerRef}
                    trainingConfiguration={trainingConfiguration}
                    defaultTrainingConfiguration={defaultTrainingConfiguration}
                    onTrainingConfigurationChange={onTrainingConfigurationChange}
                />
            ),
        },
        {
            name: 'Training',
            children: (
                <Training
                    trainingConfiguration={trainingConfiguration}
                    defaultTrainingConfiguration={defaultTrainingConfiguration}
                    onTrainingConfigurationChange={onTrainingConfigurationChange}
                />
            ),
        },
    ];

    return (
        <Tabs items={TABS} height={'100%'} UNSAFE_style={{ overflow: 'hidden' }} aria-label={'Advanced settings tabs'}>
            <TabList UNSAFE_style={{ '--spectrum-tabs-selection-indicator-color': 'var(--energy-blue)' }}>
                {(tab: TabProps) => (
                    <Item key={tab.name} textValue={tab.name}>
                        <Text>{tab.name}</Text>
                    </Item>
                )}
            </TabList>
            <TabPanels marginTop={'size-250'} UNSAFE_style={{ overflow: 'hidden' }}>
                {(tab: TabProps) => (
                    <Item key={tab.name} textValue={tab.name}>
                        <ContentWrapper ref={containerRef}>{tab.children}</ContentWrapper>
                    </Item>
                )}
            </TabPanels>
        </Tabs>
    );
};
