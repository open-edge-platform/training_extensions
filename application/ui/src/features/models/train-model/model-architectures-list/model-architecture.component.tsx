// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ModelArchitectureWithPerformanceCategory } from '../../../../constants/shared-types';
import { PerformanceCategoryBadge } from '../../model-listing/components/model-row/performance-category-badge.component';
import { ModelArchitectureCard } from './model-architecture-card/model-architecture-card.component';

type ModelArchitectureProps = {
    modelArchitecture: ModelArchitectureWithPerformanceCategory;
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
};

export const ModelArchitecture = ({
    modelArchitecture,
    selectedModelArchitectureId,
    onSelectedModelArchitectureIdChange,
}: ModelArchitectureProps) => {
    const isSelected = modelArchitecture.id === selectedModelArchitectureId;

    return (
        <ModelArchitectureCard
            modelArchitecture={modelArchitecture}
            isSelected={isSelected}
            onSelect={() => onSelectedModelArchitectureIdChange(modelArchitecture.id)}
        >
            <ModelArchitectureCard.Name />
            <ModelArchitectureCard.Parameters />

            {modelArchitecture.performanceCategory !== undefined && (
                <PerformanceCategoryBadge
                    performanceCategory={modelArchitecture.performanceCategory}
                    color={'var(--spectrum-global-color-gray-100)'}
                />
            )}
        </ModelArchitectureCard>
    );
};

export const DetailedModelArchitecture = ({
    modelArchitecture,
    selectedModelArchitectureId,
    onSelectedModelArchitectureIdChange,
}: ModelArchitectureProps) => {
    const isSelected = modelArchitecture.id === selectedModelArchitectureId;

    return (
        <ModelArchitectureCard
            modelArchitecture={modelArchitecture}
            isSelected={isSelected}
            onSelect={() => onSelectedModelArchitectureIdChange(modelArchitecture.id)}
        >
            <ModelArchitectureCard.Name />
            <ModelArchitectureCard.DetailedParameters />
        </ModelArchitectureCard>
    );
};
