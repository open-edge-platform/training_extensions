// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ModelArchitecture as ModelArchitectureType } from '../../../../constants/shared-types';
import { ModelArchitecture } from './model-architecture.component';
import { ModelArchitecturesListLayout } from './model-architectures-list-layout/model-architectures-list-layout.component';

type RecommendedModelArchitecturesProps = {
    modelArchitectures: ModelArchitectureType[];
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
};

export const RecommendedModelArchitectures = ({
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
                <ModelArchitecture
                    key={modelArchitecture.id}
                    modelArchitecture={modelArchitecture}
                    selectedModelArchitectureId={selectedModelArchitectureId}
                    onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
                />
            ))}
        </ModelArchitecturesListLayout>
    );
};
