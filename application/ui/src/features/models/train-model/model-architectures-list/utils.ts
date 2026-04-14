// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ModelArchitectureWithPerformanceCategory } from '../../../../constants/shared-types';
import { getAccuracyMetricBasedOnTask } from '../sort-model-architectures/utils';

export type AccuracyMetric = { label: string; value: number };

export const getAccuracyMetric = (
    modelArchitecture: ModelArchitectureWithPerformanceCategory
): AccuracyMetric | undefined => {
    const value = getAccuracyMetricBasedOnTask(modelArchitecture);

    if (value === null || value === undefined) {
        return undefined;
    }

    const label = modelArchitecture.task === 'classification' ? 'Top-1 Acc' : 'mAP';

    return { label, value };
};

const getRecommendedArchitectures = (modelArchitectures: ModelArchitectureWithPerformanceCategory[]) => {
    const recommended = modelArchitectures.filter(
        (modelArchitecture) => modelArchitecture.performanceCategory !== undefined
    );

    if (recommended.length > 0) {
        return recommended;
    }

    // For now just return top 3 recommended architectures, but in the future we can add more logic here
    return modelArchitectures.slice(0, 3);
};

export const getRecommendedModelArchitecturesWithActiveArchitecture = (
    modelArchitectures: ModelArchitectureWithPerformanceCategory[],
    activeModelArchitectureId: string | undefined
) => {
    const recommended = getRecommendedArchitectures(modelArchitectures);

    const foundActiveArchitectureInRecommended = recommended.find(
        (modelArchitecture) => modelArchitecture.id === activeModelArchitectureId
    );

    if (foundActiveArchitectureInRecommended !== undefined) {
        return recommended;
    }

    const activeModelArchitecture = modelArchitectures.find(
        (modelArchitecture) => modelArchitecture.id === activeModelArchitectureId
    );

    if (activeModelArchitecture === undefined) {
        return recommended;
    }

    return [activeModelArchitecture, ...recommended];
};
