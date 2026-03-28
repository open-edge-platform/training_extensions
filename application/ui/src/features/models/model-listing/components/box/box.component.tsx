// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Content, Flex, Heading } from '@geti/ui';
import { clsx } from 'clsx';

import classes from './box.module.scss';

type BoxProps = {
    title: ReactNode;
    content: ReactNode;
    headingClassName?: string;
    contentClassName?: string;
    testId?: string;
};

export const Box = ({ title, content, headingClassName, contentClassName, testId }: BoxProps) => {
    return (
        <Flex direction={'column'} height={'100%'} UNSAFE_className={classes.boxWrapper} data-testid={testId}>
            <Heading UNSAFE_className={clsx(classes.boxHeading, headingClassName)} level={5}>
                {title}
            </Heading>
            <Content UNSAFE_className={clsx(classes.boxContent, contentClassName)}>{content}</Content>
        </Flex>
    );
};
