// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Flex, Grid } from '@geti/ui';

import classes from '../media-preview.module.scss';

type SecondaryToolbarProps = {
    content: ReactNode;
    isHidden: boolean;
};
export const SecondaryToolbar = ({ content, isHidden }: SecondaryToolbarProps) => {
    return (
        <Flex height={'100%'} alignItems={'center'} isHidden={isHidden} margin={'size-100'}>
            <Grid UNSAFE_className={classes.toolbarGrid}>
                <Flex UNSAFE_className={classes.toolbarSection}>{content}</Flex>
            </Grid>
        </Flex>
    );
};
