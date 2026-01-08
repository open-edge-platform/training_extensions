// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Disclosure, DisclosurePanel, DisclosureTitle, Item, TabList, TabPanels, Tabs, Text } from '@geti/ui';

import type { SchemaModelView } from '../../../api/openapi-spec';
import { GroupHeader } from './group-headers/group-header.component';
import { ModelRow } from './model-row.component';
import { ModelVariantsTabs } from './model-variants/model-variant-tabs.component';
import { ModelsTableHeader } from './models-table-header.component';
import type { ArchitectureGroup, DatasetGroup, GroupByMode } from './types';

import classes from './model-listing.module.scss';

interface GroupModelsContainerProps {
    groupBy: GroupByMode;
    group: DatasetGroup | ArchitectureGroup;
    models: SchemaModelView[];
}

export const GroupModelsContainer = ({ groupBy, group, models }: GroupModelsContainerProps) => {
    return (
        <>
            <GroupHeader groupBy={groupBy} data={group} />
            <ModelsTableHeader groupBy={groupBy} />

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
        </>
    );
};
