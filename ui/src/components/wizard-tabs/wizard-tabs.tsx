// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ComponentProps } from 'react';

import { Tab as AriaTab, TabList, Tabs } from 'react-aria-components';

import { WizardState } from './interfaces';

import classes from './wizard-tabs.module.scss';

export const Tab = ({ name, ...props }: { name: string } & ComponentProps<typeof AriaTab>) => {
    return <AriaTab {...props}>{name}</AriaTab>;
};

export const WizardTabs = ({
    state,
    pathname,
    content,
}: {
    state: WizardState;
    pathname: string;
    content: JSX.Element;
}) => {
    return (
        <Tabs selectedKey={pathname}>
            <TabList aria-label='Pipeline' className={classes.tabList}>
                {state.map((tab) => {
                    return <Tab key={tab.href} id={tab.href} href={tab.href} name={tab.name} className={classes.tab} />;
                })}
            </TabList>

            {content}
        </Tabs>
    );
};
