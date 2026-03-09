// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';

import type { ModelArchitecture as ModelArchitectureType } from '../../../../constants/shared-types';
import { SORTING_HANDLERS, SortingOptions } from '../sort-model-architectures/utils';
import { ModelArchitecture } from './model-architecture.component';
import { ModelArchitecturesListLayout } from './model-architectures-list-layout/model-architectures-list-layout.component';

type AllModelArchitecturesProps = {
    activeModelArchitectureId: string | undefined;
    modelArchitectures: ModelArchitectureType[];
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
    sortBy: SortingOptions;
};

export const AllModelArchitectures = ({
    activeModelArchitectureId,
    modelArchitectures,
    onSelectedModelArchitectureIdChange,
    selectedModelArchitectureId,
    sortBy,
}: AllModelArchitecturesProps) => {
    const sortedModelArchitectures = SORTING_HANDLERS[sortBy](modelArchitectures);

    return (
        <Flex direction={'column'} gap={'size-200'}>
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
