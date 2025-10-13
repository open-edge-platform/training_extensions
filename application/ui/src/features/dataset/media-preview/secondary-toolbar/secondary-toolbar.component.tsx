// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Flex, Grid } from '@geti/ui';

export const SecondaryToolbar = ({ content }: { content: ReactNode }) => {
    return (
        <Flex height={'100%'} alignItems={'center'} justifyContent={'center'} margin={'size-100'}>
            <Grid>
                <Flex>Something</Flex>
                <Flex>{content}</Flex>
            </Grid>
        </Flex>
    );
};
