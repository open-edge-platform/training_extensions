// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';

import { EmptySearchResults } from './components/expandable-search/empty-search-results.component';
import { GroupModelsContainer } from './components/group-models-container/group-models-container.component';
import { useModelListing } from './provider/model-listing-provider';

export const ModelListing = () => {
    const { groupedModels, searchBy } = useModelListing();

    const hasNoResults = groupedModels.length === 0 && searchBy.length > 0;

    return (
        <>
            {hasNoResults ? (
                <Flex direction={'column'} flex={1}>
                    <EmptySearchResults />
                </Flex>
            ) : (
                <Flex direction={'column'} gap={'size-300'} flex={1}>
                    {groupedModels.map(({ group, models }, index) => (
                        <GroupModelsContainer
                            key={'id' in group ? group.id : `${group.name}-${index}`}
                            group={group}
                            models={models}
                        />
                    ))}
                </Flex>
            )}
        </>
    );
};
