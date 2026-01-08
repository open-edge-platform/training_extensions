// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Item, TabList, TabPanels, Tabs, Text } from '@geti/ui';

import type { SchemaModelView } from '../../../../api/openapi-spec';
import { ModelVariantsTabs } from '../model-variants/model-variant-tabs.component';

interface ModelDetailsTabsProps {
    model: SchemaModelView;
}

export const ModelDetailsTabs = ({ model }: ModelDetailsTabsProps) => {
    return (
        <Tabs aria-label='Model details'>
            <TabList>
                <Item key='variants'>
                    <Text>Model variants</Text>
                </Item>
                <Item key='metrics'>
                    <Text>Model metrics</Text>
                </Item>
                <Item key='parameters'>
                    <Text>Training parameters</Text>
                </Item>
                <Item key='datasets'>
                    <Text>Training datasets</Text>
                </Item>
            </TabList>
            <TabPanels>
                <Item key='variants'>
                    <ModelVariantsTabs model={model} />
                </Item>
                <Item key='metrics'>
                    <Text>Model metrics content</Text>
                </Item>
                <Item key='parameters'>
                    <Text>Training parameter settings content</Text>
                </Item>
                <Item key='datasets'>
                    <Text>Training datasets content</Text>
                </Item>
            </TabPanels>
        </Tabs>
    );
};
