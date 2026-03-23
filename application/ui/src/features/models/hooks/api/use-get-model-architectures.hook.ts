// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { usePrefetchQuery, useSuspenseQuery } from '@tanstack/react-query';
import { useProject } from 'hooks/api/project.hook';

import { $api } from '../../../../api/client';
import {
    ModelArchitecture,
    ModelArchitectureWithPerformanceCategory,
    RecommendedModelArchitectures,
    TaskType,
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

const getTaskModelArchitecturesQueryOptions = (taskType: TaskType) => {
    return $api.queryOptions('get', '/api/model_architectures', {
        params: { query: { task: taskType } },
    });
};

export const usePrefetchTaskModelArchitectures = () => {
    const { data: projectData } = useProject();

    return usePrefetchQuery(getTaskModelArchitecturesQueryOptions(projectData.task.task_type));
};

export const useGetTaskModelArchitectures = () => {
    const { data: projectData } = useProject();

    const { data } = useSuspenseQuery(getTaskModelArchitecturesQueryOptions(projectData.task.task_type));

    const modelArchitectures = useMemo(
        () => getModelArchitectures(data.model_architectures, data.top_picks),
        [data.model_architectures, data.top_picks]
    );

    return {
        modelArchitectures,
    };
};
