// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Disclosure, DisclosurePanel, DisclosureTitle, Flex } from '@geti/ui';

import { SchemaModelView } from '../../../../api/openapi-spec';
import { ModelDetailsTabs } from '../model-details/model-details-tabs.component';
import { useModelListing } from '../provider/model-listing-provider';
import { ArchitectureGroup, DatasetGroup } from '../types';
import { GroupHeader } from './group-headers/group-header.component';
import { ModelRow } from './model-row.component';
import { ModelsTableHeader } from './models-table-header.component';

import classes from './group-models-container.module.scss';

interface GroupModelsContainerProps {
    group: DatasetGroup | ArchitectureGroup;
    models: SchemaModelView[];
}

export const GroupModelsContainer = ({ group, models }: GroupModelsContainerProps) => {
    const { expandedModelIds, onExpandModel } = useModelListing();

    return (
        <Flex direction={'column'} UNSAFE_className={classes.groupModelsContainer}>
            <GroupHeader data={group} />
            <ModelsTableHeader />

            {models.map((model) => (
                <Disclosure
                    key={model.id}
                    isQuiet
                    UNSAFE_className={classes.disclosure}
                    isExpanded={model.id ? expandedModelIds.has(model.id) : false}
                    onExpandedChange={() => model.id && onExpandModel(model.id)}
                >
                    <DisclosureTitle UNSAFE_className={classes.disclosureItem}>
                        <ModelRow model={model} />
                    </DisclosureTitle>
                    <DisclosurePanel>
                        <ModelDetailsTabs model={model} />
                    </DisclosurePanel>
                </Disclosure>
            ))}
        </Flex>
    );
};
