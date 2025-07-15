import { ComponentProps, ReactNode } from 'react';

import { Text } from '@geti/ui';
import { Checkmark } from '@geti/ui/icons';
import { Tab, TabList, Tabs } from 'react-aria-components';

import classes from './wizard-steps.module.scss';

export const WizardTab = ({
    number,
    isCompleted = false,
    children,
    ...props
}: ComponentProps<typeof Tab> & {
    number: number;
    isCompleted?: boolean;
    children: ReactNode;
}) => {
    return (
        <Tab {...props} className={classes.wizardTab}>
            <Text UNSAFE_className={classes.stepNumber}>{isCompleted ? <Checkmark size='S' /> : number}</Text>
            <Text UNSAFE_className={classes.wizardTabText}>{children}</Text>
        </Tab>
    );
};

export const WizardTabs = (props: ComponentProps<typeof Tabs>) => {
    return <Tabs {...props} className={classes.wizardTabs} />;
};

export const WizardTabList = (props: ComponentProps<typeof TabList>) => {
    return <TabList {...props} className={classes.tabList} />;
};
