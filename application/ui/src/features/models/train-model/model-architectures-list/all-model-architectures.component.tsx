// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Flex } from '@geti/ui';

import type { ModelArchitecture as ModelArchitectureType } from '../../../../constants/shared-types';
import { SortModelArchitectures } from '../sort-model-architectures/sort-model-architectures.component';
import { SORT_OPTIONS, SORTING_HANDLERS, SortingOptions } from '../sort-model-architectures/utils';
import { ModelArchitectureCard } from './model-architecture/model-architecture.component';
import { ModelArchitecturesListLayout } from './model-architectures-list-layout/model-architectures-list-layout.component';

type ModelArchitectureProps = {
    activeModelArchitectureId: string | undefined;
    modelArchitecture: ModelArchitectureType;
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
};

const ModelArchitecture = ({
    activeModelArchitectureId,
    modelArchitecture,
    onSelectedModelArchitectureIdChange,
    selectedModelArchitectureId,
}: ModelArchitectureProps) => {
    const isSelected = modelArchitecture.id === selectedModelArchitectureId;
    const isActive = activeModelArchitectureId === modelArchitecture.id;

    return (
        <ModelArchitectureCard
            isCompact
            modelArchitecture={modelArchitecture}
            isSelected={isSelected}
            onSelect={() => onSelectedModelArchitectureIdChange(modelArchitecture.id)}
        >
            <Flex direction={'column'} width={'100%'} minWidth={0} gap={'size-100'}>
                {isActive && <ModelArchitectureCard.Active />}
                <Flex alignItems={'center'} justifyContent={'space-between'}>
                    <ModelArchitectureCard.Name />
                    <ModelArchitectureCard.Description />
                </Flex>
            </Flex>
            <ModelArchitectureCard.Parameters />
        </ModelArchitectureCard>
    );
};

type AllModelArchitecturesProps = {
    activeModelArchitectureId: string | undefined;
    modelArchitectures: ModelArchitectureType[];
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
};

export const AllModelArchitectures = ({
    activeModelArchitectureId,
    modelArchitectures,
    onSelectedModelArchitectureIdChange,
    selectedModelArchitectureId,
}: AllModelArchitecturesProps) => {
    const [sortBy, setSortBy] = useState<SortingOptions>(SortingOptions.RELEVANCE_ASC);
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
                    <ModelArchitecture
                        key={modelArchitecture.id}
                        activeModelArchitectureId={activeModelArchitectureId}
                        modelArchitecture={modelArchitecture}
                        selectedModelArchitectureId={selectedModelArchitectureId}
                        onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
                    />
                ))}
            </ModelArchitecturesListLayout>
        </Flex>
    );
};
