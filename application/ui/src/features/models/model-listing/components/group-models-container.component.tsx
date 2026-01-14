// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Disclosure, DisclosurePanel, DisclosureTitle, Flex } from '@geti/ui';

import { SchemaModelView } from '../../../../api/openapi-spec';
import { ModelDetailsTabs } from '../model-details/model-details-tabs.component';
import { ArchitectureGroup, DatasetGroup, GroupByMode, SortBy } from '../types';
import { GroupHeader } from './group-headers/group-header.component';
import { ModelRow } from './model-row.component';
import { ModelsTableHeader } from './models-table-header.component';

import classes from './group-models-container.module.scss';

interface GroupModelsContainerProps {
    groupBy: GroupByMode;
    group: DatasetGroup | ArchitectureGroup;
    models: SchemaModelView[];
    sortBy?: SortBy;
    expandedModelIds: Set<string>;
    onExpandModel: (modelId: string) => void;
}

export const GroupModelsContainer = ({
    groupBy,
    group,
    models,
    sortBy,
    expandedModelIds,
    onExpandModel,
}: GroupModelsContainerProps) => {
    return (
        <Flex direction={'column'} UNSAFE_className={classes.datasetWrapper}>
            <GroupHeader groupBy={groupBy} data={group} />
            <ModelsTableHeader groupBy={groupBy} sortBy={sortBy} />

            {models.map((model) => (
                <Disclosure
                    key={model.id}
                    isQuiet
                    UNSAFE_className={classes.disclosure}
                    isExpanded={model.id ? expandedModelIds.has(model.id) : false}
                    onExpandedChange={() => model.id && onExpandModel(model.id)}
                >
                    <DisclosureTitle UNSAFE_className={classes.disclosureItem}>
                        <ModelRow model={model} onExpandModel={onExpandModel} />
                    </DisclosureTitle>
                    <DisclosurePanel>
                        <ModelDetailsTabs model={model} />
                    </DisclosurePanel>
                </Disclosure>
            ))}
        </Flex>
    );
};
