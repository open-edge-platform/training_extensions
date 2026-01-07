// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    Disclosure,
    DisclosurePanel,
    DisclosureTitle,
    Divider,
    Item,
    TabList,
    TabPanels,
    Tabs,
    Text,
    View,
} from '@geti/ui';

import { DatasetHeaderRow, DatasetItem } from './dataset-item/dataset-item.component';
import { Header } from './header.component';
import { ModelRow } from './model-row.component';
import { ModelVariantsTabs } from './model-variants/model-variant-tabs.component';

import classes from './model-listing.module.scss';

// TODO: Replace with dynamic data
const models = [
    { id: 1, name: 'Model Project #1' },
    { id: 2, name: 'Model Project #2' },
];

export const ModelListing = () => {
    return (
        <View padding={'size-300'}>
            <Header />

            <Divider size={'S'} marginY={'size-300'} />

            {/* TODO: Update to a more generic name, since this will either be a dataset, or a model */}
            <DatasetItem />

            <DatasetHeaderRow />

            {/* 
                TODO: Update this to a dynamic value.
                It will either be all models from the dataset, or all models from an architecture
            */}
            {models.map((model) => (
                <Disclosure key={model.id} isQuiet UNSAFE_className={classes.disclosure}>
                    <DisclosureTitle UNSAFE_className={classes.disclosureItem}>
                        <ModelRow model={model} />
                    </DisclosureTitle>
                    <DisclosurePanel>
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
                                    <ModelVariantsTabs />
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
                    </DisclosurePanel>
                </Disclosure>
            ))}
        </View>
    );
};
