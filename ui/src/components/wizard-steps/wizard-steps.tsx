// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ComponentProps, ReactNode } from 'react';

import { Text } from '@geti/ui';
import { Checkmark } from '@geti/ui/icons';
import { Tab, TabList, Tabs } from 'react-aria-components';

import { WizardState } from './interfaces';

import classes from './wizard-steps.module.scss';

export const Step = ({
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
        <Tab {...props} className={classes.wizardStep}>
            <Text UNSAFE_className={classes.stepNumber}>{isCompleted ? <Checkmark size='S' /> : number}</Text>
            <Text UNSAFE_className={classes.wizardStepText}>{children}</Text>
        </Tab>
    );
};

export const Steps = (props: ComponentProps<typeof Tabs>) => {
    return <Tabs {...props} className={classes.wizardSteps} />;
};

export const StepList = (props: ComponentProps<typeof TabList>) => {
    return <TabList {...props} className={classes.tabList} />;
};

export const WizardSteps = ({
    state,
    pathname,
    content,
}: {
    state: WizardState;
    pathname: string;
    content: ReactNode;
}) => {
    return (
        <Steps
            selectedKey={pathname}
            style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                width: '100%',
            }}
        >
            <StepList aria-label='Pipeline' style={{ flexGrow: '1', width: '100%' }}>
                {state.map((step, idx) => {
                    return (
                        <Step
                            key={step.href}
                            id={step.href}
                            href={step.href}
                            number={idx + 1}
                            isDisabled={step.isDisabled}
                            isCompleted={step.isCompleted}
                        >
                            {step.name}
                        </Step>
                    );
                })}
            </StepList>
            {content}
        </Steps>
    );
};
