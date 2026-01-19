// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';

import type { ModelArchitecture as ModelArchitectureType } from '../../../../constants/shared-types';
import { ModelArchitectureCard } from './model-architecture/model-architecture.component';
import { ModelArchitecturesListLayout } from './model-architectures-list-layout/model-architectures-list-layout.component';

type RecommendedModelArchitectureProps = {
    activeModelArchitectureId: string | undefined;
    modelArchitecture: ModelArchitectureType;
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
};

const RecommendedModelArchitecture = ({
    activeModelArchitectureId,
    modelArchitecture,
    selectedModelArchitectureId,
    onSelectedModelArchitectureIdChange,
}: RecommendedModelArchitectureProps) => {
    const isSelected = modelArchitecture.id === selectedModelArchitectureId;
    const isActive = activeModelArchitectureId === modelArchitecture.id;

    return (
        <ModelArchitectureCard
            modelArchitecture={modelArchitecture}
            isSelected={isSelected}
            onSelect={() => onSelectedModelArchitectureIdChange(modelArchitecture.id)}
        >
            <Flex width={'100%'} minWidth={0} direction={'column'} gap={'size-100'}>
                {isActive && <ModelArchitectureCard.Active />}
                <ModelArchitectureCard.Name />
            </Flex>
            <ModelArchitectureCard.Parameters />
            <ModelArchitectureCard.Divider />
            <ModelArchitectureCard.ExpandedDescription />
        </ModelArchitectureCard>
    );
};

type RecommendedModelArchitecturesProps = {
    activeModelArchitectureId: string | undefined;
    modelArchitectures: ModelArchitectureType[];
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
};

export const RecommendedModelArchitectures = ({
    activeModelArchitectureId,
    modelArchitectures,
    onSelectedModelArchitectureIdChange,
    selectedModelArchitectureId,
}: RecommendedModelArchitecturesProps) => {
    return (
        <ModelArchitecturesListLayout
            selectedModelArchitectureId={selectedModelArchitectureId}
            onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
            ariaLabel={'Recommended model architectures'}
        >
            {modelArchitectures.map((modelArchitecture) => (
                <RecommendedModelArchitecture
                    key={modelArchitecture.id}
                    activeModelArchitectureId={activeModelArchitectureId}
                    modelArchitecture={modelArchitecture}
                    selectedModelArchitectureId={selectedModelArchitectureId}
                    onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
                />
            ))}
        </ModelArchitecturesListLayout>
    );
};
