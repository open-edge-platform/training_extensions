// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid, Item, Picker, Text } from '@geti/ui';

export const ModelsHeader = () => {
    return (
        <Grid columns={['auto auto 1fr']} alignItems={'center'}>
            <Text>Models</Text>

            <Flex marginStart={'size-300'} gap={'size-100'}>
                <Picker aria-label='Group by'>
                    <Item key='dataset'>Grouped by: Dataset</Item>
                    <Item key='architecture'>Grouped by: Architecture</Item>
                </Picker>
                <Picker aria-label='Sort by'>
                    <Item key='active-model'>Sort: Active model</Item>
                    <Item key='architecture'>Sort: Architecture</Item>
                </Picker>
            </Flex>
        </Grid>
    );
};
