// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid } from '@geti/ui';

export const SecondaryToolbar = () => {
    return (
        <Flex height={'100%'} alignItems={'center'} justifyContent={'center'} margin={'size-100'}>
            <Grid>
                <Flex>Something</Flex>
                <Flex>Something else</Flex>
            </Grid>
        </Flex>
    );
};
