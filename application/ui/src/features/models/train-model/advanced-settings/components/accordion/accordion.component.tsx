// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ComponentProps, ReactNode } from 'react';

import { Disclosure, DisclosurePanel, DisclosureTitle, Divider, Flex, Text, View } from '@geti/ui';
import { AlertOutlined } from '@geti/ui/icons';
import { clsx } from 'clsx';

import classes from './accordion.module.scss';

type DisclosureProps = Omit<ComponentProps<typeof Disclosure>, 'isQuiet' | 'defaultExpanded'>;
type DisclosureTitleProps = ComponentProps<typeof DisclosureTitle>;
type DisclosurePanelProps = ComponentProps<typeof DisclosurePanel>;

interface AccordionTag {
    children: ReactNode;
    ariaLabel?: string;
}

const AccordionTitle = ({ UNSAFE_className, ...props }: DisclosureTitleProps) => {
    return <DisclosureTitle {...props} UNSAFE_className={clsx(UNSAFE_className, classes.accordionTitle)} />;
};

const AccordionContent = ({ UNSAFE_className, ...props }: DisclosurePanelProps) => {
    return <DisclosurePanel {...props} UNSAFE_className={clsx(UNSAFE_className, classes.accordionContent)} />;
};

const AccordionTag = ({ children, ariaLabel }: AccordionTag) => {
    return (
        <View borderRadius={'regular'} borderWidth={'thin'} padding={'size-50'} UNSAFE_className={classes.accordionTag}>
            <div aria-label={ariaLabel}>{children}</div>
        </View>
    );
};

const AccordionDivider = (props: Omit<ComponentProps<typeof Divider>, 'size'>) => {
    return <Divider size={'S'} {...props} />;
};

const AccordionDescription = ({ children }: { children: ReactNode }) => {
    return <Text UNSAFE_className={classes.accordionDescription}>{children}</Text>;
};

const AccordionWarning = ({ children }: { children: ReactNode }) => {
    return (
        <Flex alignItems={'center'} gap={'size-100'} UNSAFE_className={classes.warning}>
            <AlertOutlined />
            {children}
        </Flex>
    );
};

export const Accordion = ({ UNSAFE_className, ...props }: DisclosureProps) => {
    return (
        <Disclosure isQuiet defaultExpanded {...props} UNSAFE_className={clsx(UNSAFE_className, classes.accordion)} />
    );
};

Accordion.Title = AccordionTitle;
Accordion.Content = AccordionContent;
Accordion.Tag = AccordionTag;
Accordion.Divider = AccordionDivider;
Accordion.Description = AccordionDescription;
Accordion.Warning = AccordionWarning;
