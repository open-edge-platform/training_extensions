// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Flex } from '@geti/ui';

import type { ModelArchitecture as ModelArchitectureType } from '../../../../constants/shared-types';
import { SortModelArchitectures } from '../sort-model-architectures/sort-model-architectures.component';
import { SORT_OPTIONS, SORTING_HANDLERS, SortingOptions } from '../sort-model-architectures/utils';
import { DetailedModelArchitecture } from './model-architecture.component';
import { ModelArchitecturesListLayout } from './model-architectures-list-layout/model-architectures-list-layout.component';

type AllModelArchitecturesProps = {
    modelArchitectures: ModelArchitectureType[];
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
};

export const AllModelArchitectures = ({
    modelArchitectures,
    onSelectedModelArchitectureIdChange,
    selectedModelArchitectureId,
}: AllModelArchitecturesProps) => {
    const [sortBy, setSortBy] = useState<SortingOptions>(SortingOptions.NAME_ASC);
    const sortedModelArchitectures = SORTING_HANDLERS[sortBy](modelArchitectures);

    return (
        <Flex direction={'column'} gap={'size-200'}>
            <SortModelArchitectures sortBy={sortBy} onSort={setSortBy} items={SORT_OPTIONS} />
            <ModelArchitecturesListLayout
                selectedModelArchitectureId={selectedModelArchitectureId}
                onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
                ariaLabel={'ALL model architectures'}
            >
                {sortedModelArchitectures.map((modelArchitecture) => (
                    <DetailedModelArchitecture
                        key={modelArchitecture.id}
                        modelArchitecture={modelArchitecture}
                        selectedModelArchitectureId={selectedModelArchitectureId}
                        onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
                    />
                ))}
            </ModelArchitecturesListLayout>
        </Flex>
    );
};
