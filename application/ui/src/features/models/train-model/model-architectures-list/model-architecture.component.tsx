// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, View } from '@geti/ui';

import type { ModelArchitectureWithPerformanceCategory } from '../../../../constants/shared-types';
import { PerformanceCategoryBadge } from '../../model-listing/components/model-row/performance-category-badge.component';
import { ModelArchitectureCard } from './model-architecture-card/model-architecture-card.component';

type ModelArchitectureProps = {
    activeModelArchitectureId: string | undefined;
    modelArchitecture: ModelArchitectureWithPerformanceCategory;
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
    showBenchmarkStats?: boolean;
};

export const ModelArchitecture = ({
    activeModelArchitectureId,
    modelArchitecture,
    selectedModelArchitectureId,
    onSelectedModelArchitectureIdChange,
    showBenchmarkStats = false,
}: ModelArchitectureProps) => {
    const isSelected = modelArchitecture.id === selectedModelArchitectureId;
    const isActive = activeModelArchitectureId === modelArchitecture.id;

    return (
        <ModelArchitectureCard
            modelArchitecture={modelArchitecture}
            isSelected={isSelected}
            onSelect={() => onSelectedModelArchitectureIdChange(modelArchitecture.id)}
        >
            <ModelArchitectureCard.Name />
            {showBenchmarkStats ? <ModelArchitectureCard.DetailedParameters /> : <ModelArchitectureCard.Parameters />}

            {(isActive || (!showBenchmarkStats && modelArchitecture.performanceCategory !== undefined)) && (
                <Flex gap={'size-100'} alignItems={'center'}>
                    {isActive && (
                        <View justifySelf={'start'}>
                            <ModelArchitectureCard.Active />
                        </View>
                    )}
                    {!showBenchmarkStats && modelArchitecture.performanceCategory !== undefined && (
                        <PerformanceCategoryBadge
                            performanceCategory={modelArchitecture.performanceCategory}
                            color={'var(--spectrum-global-color-gray-100)'}
                        />
                    )}
                </Flex>
            )}
        </ModelArchitectureCard>
    );
};
