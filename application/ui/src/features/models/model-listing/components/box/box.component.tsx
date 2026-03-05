// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Content, Flex, Heading } from '@geti/ui';

import classes from './box.module.scss';

type BoxProps = {
    title: string;
    content: ReactNode;
    customClasses?: string;
};

export const Box = ({ title, content, customClasses }: BoxProps) => {
    return (
        <Flex direction={'column'} height={'100%'} UNSAFE_className={customClasses}>
            <Heading UNSAFE_className={classes.boxHeading} level={5}>
                {title}
            </Heading>
            <Content UNSAFE_className={classes.boxContent}>{content}</Content>
        </Flex>
    );
};
