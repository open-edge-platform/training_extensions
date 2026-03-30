// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { View } from '@geti-ui/ui';

import type { ModelArchitecture as ModelArchitectureType } from '../../../../constants/shared-types';
import { ModelArchitectureCard } from './model-architecture-card/model-architecture-card.component';

type ModelArchitectureProps = {
    activeModelArchitectureId: string | undefined;
    modelArchitecture: ModelArchitectureType;
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
};

export const ModelArchitecture = ({
    activeModelArchitectureId,
    modelArchitecture,
    selectedModelArchitectureId,
    onSelectedModelArchitectureIdChange,
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
            <ModelArchitectureCard.Parameters />
            {isActive && (
                <View justifySelf={'start'}>
                    <ModelArchitectureCard.Active />
                </View>
            )}
        </ModelArchitectureCard>
    );
};
