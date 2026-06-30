// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Flex, Grid, Heading, ToggleButton, Tooltip, TooltipTrigger, View } from '@geti-ui/ui';
import { Gear, GraphChart } from '@geti-ui/ui/icons';

import { ReactComponent as PipelineIcon } from '../../../assets/icons/pipeline.svg';
import { DataCollection } from './data-collection.component';
import { Graphs } from './graphs.component';
import { PipelineConfiguration } from './pipeline-configuration.component';

import styles from './sidebar-tabs.module.scss';

const TABS = [
    { label: 'Pipeline configuration', icon: <PipelineIcon />, content: <PipelineConfiguration /> },
    { label: 'Data collection policy', icon: <Gear />, content: <DataCollection /> },
    { label: 'Model statistics', icon: <GraphChart />, content: <Graphs /> },
];

type TabProps = {
    tabs: (typeof TABS)[number][];
    selectedTab: string;
};

const SidebarTabs = ({ tabs, selectedTab }: TabProps) => {
    const [tab, setTab] = useState<string | null>(selectedTab);

    const isExpanded = tab !== null;
    const gridTemplateColumns = isExpanded ? ['clamp(size-4600, 30vw, 40rem)', 'size-600'] : ['0px', 'size-600'];

    const content = tabs.find(({ label }) => label === tab)?.content;

    const handleSetTab = (label: string) => {
        setTab((prev) => (prev === label ? null : label));
    };

    return (
        <Grid
            gridArea={'aside'}
            UNSAFE_className={styles.container}
            columns={gridTemplateColumns}
            data-expanded={isExpanded}
            minHeight={0}
        >
            <View
                gridColumn={'1/2'}
                UNSAFE_className={styles.sidebarContent}
                backgroundColor={'gray-100'}
                paddingY={'size-400'}
                paddingX={'size-500'}
                aria-hidden={!isExpanded || undefined}
            >
                {isExpanded && (
                    <>
                        <Flex alignItems='center' gap={'size-100'} marginBottom={'size-300'}>
                            <Heading level={2}>{tab}</Heading>
                        </Flex>
                        <Flex direction={'column'} flex={1} UNSAFE_style={{ overflow: 'hidden auto' }}>
                            {content}
                        </Flex>
                    </>
                )}
            </View>
            <View gridColumn={'2/3'} backgroundColor={'gray-200'} padding={'size-100'}>
                <Flex direction={'column'} height={'100%'} alignItems={'center'} gap={'size-100'}>
                    {tabs.map(({ label, icon }) => (
                        <TooltipTrigger key={label} placement={'left'}>
                            <ToggleButton
                                isQuiet
                                isSelected={label === tab}
                                onChange={() => handleSetTab(label)}
                                UNSAFE_className={styles.toggleButton}
                                aria-label={`Toggle ${label} tab`}
                            >
                                {icon}
                            </ToggleButton>
                            <Tooltip>{label}</Tooltip>
                        </TooltipTrigger>
                    ))}
                </Flex>
            </View>
        </Grid>
    );
};

export const Sidebar = () => {
    return <SidebarTabs tabs={TABS} selectedTab={TABS[0].label} />;
};
