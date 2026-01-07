// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid, Item, Picker, Text, ToggleButton } from '@geti/ui';

export const Header = () => {
    return (
        <Grid columns={['auto auto 1fr']} gap={'size-100'} alignItems={'center'}>
            <Text>Models</Text>

            <Flex marginStart={'size-300'} gap={'size-100'}>
                <Picker
                    placeholder={'Group by'}
                    width={'size-3000'}
                    aria-label={'Group models'}
                    defaultSelectedKey={'dataset'}
                >
                    <Item key='dataset'>Group by: Dataset</Item>
                    <Item key='architecture'>Group by: Architecture</Item>
                </Picker>
                <Picker
                    placeholder={'Sort by'}
                    width={'size-3000'}
                    aria-label={'Sort models'}
                    defaultSelectedKey={'active-model'}
                >
                    <Item key='active-model'>Sort: Active model</Item>
                    <Item key='architecture'>Sort: Architecture</Item>
                </Picker>
            </Flex>

            <Flex>
                <ToggleButton isEmphasized>Pin active model on top</ToggleButton>
            </Flex>
        </Grid>
    );
};
