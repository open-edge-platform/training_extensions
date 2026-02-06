// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { useProject } from 'hooks/api/project.hook';

import { $api } from '../../../../api/client';
import {
    ModelArchitecture,
    ModelArchitectureWithPerformanceCategory,
    RecommendedModelArchitectures,
} from '../../../../constants/shared-types';

const getModelArchitectures = (
    modelArchitectures: ModelArchitecture[],
    recommendedModelArchitectures: RecommendedModelArchitectures | null
): ModelArchitectureWithPerformanceCategory[] => {
    if (recommendedModelArchitectures === null) {
        return modelArchitectures;
    }

    // Recommended architectures have the shape like { balance: "id-1", speed: "id-2", accuracy: "id-3" }
    // Here we need to convert it to { "id-1": "balance", "id-2": "speed", "id-3": "accuracy" }
    const recommendedArchitectureIdToCategory = Object.fromEntries(
        Object.entries(recommendedModelArchitectures).map(([key, value]) => [value, key])
    );

    return modelArchitectures.map((modelArchitecture) => {
        if (recommendedArchitectureIdToCategory[modelArchitecture.id] === undefined) {
            return modelArchitecture;
        }

        return {
            ...modelArchitecture,
            performanceCategory: recommendedArchitectureIdToCategory[modelArchitecture.id],
        };
    });
};

export const useGetTaskModelArchitectures = () => {
    const { data: projectData } = useProject();

    const { data } = $api.useSuspenseQuery('get', '/api/model_architectures', {
        params: {
            query: {
                task: projectData.task.task_type,
            },
        },
    });

    const modelArchitectures = useMemo(
        () => getModelArchitectures(data.model_architectures, data.top_picks),
        [data.model_architectures, data.top_picks]
    );

    return {
        modelArchitectures,
    };
};
