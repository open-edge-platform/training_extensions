// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider, Flex, View } from '@geti/ui';

import { EmptySearchResults } from './components/empty-search-results.component';
import { GroupModelsContainer } from './components/group-models-container.component';
import { Header } from './components/header.component';
import { useModelListing } from './provider/model-listing-provider';

export const ModelListing = () => {
    const { groupedModels, searchBy } = useModelListing();

    const hasNoResults = groupedModels.length === 0 && searchBy.length > 0;

    return (
        <View padding={'size-300'}>
            <Header />

            <Divider size={'S'} marginY={'size-300'} />

            {hasNoResults ? (
                <EmptySearchResults />
            ) : (
                <Flex direction={'column'} gap={'size-300'}>
                    {groupedModels.map(({ group, models }, index) => (
                        <GroupModelsContainer
                            key={'id' in group ? group.id : `${group.name}-${index}`}
                            group={group}
                            models={models}
                        />
                    ))}
                </Flex>
            )}
        </View>
    );
};
