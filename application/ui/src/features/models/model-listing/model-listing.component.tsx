// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti-ui/ui';

import { EmptySearchResults } from './components/expandable-search/empty-search-results.component';
import { GroupModelsContainer } from './components/group-models-container/group-models-container.component';
import { GroupedModels } from './types';

type ModelListingProps = {
    hasNoResults: boolean;
    groupedModels: GroupedModels[];
};

export const ModelListing = ({ groupedModels, hasNoResults }: ModelListingProps) => {
    return (
        <>
            {hasNoResults ? (
                <Flex direction={'column'} flex={1}>
                    <EmptySearchResults />
                </Flex>
            ) : (
                <Flex
                    direction={'column'}
                    flex={1}
                    UNSAFE_style={{ backgroundColor: 'var(--spectrum-global-color-gray-50)' }}
                >
                    {groupedModels.map(({ group, models }) => (
                        <GroupModelsContainer key={group.id} group={group} models={models} />
                    ))}
                </Flex>
            )}
        </>
    );
};
