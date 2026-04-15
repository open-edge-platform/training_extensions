// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isNil } from 'lodash-es';

import type { BenchmarkMetrics, ModelArchitectureWithPerformanceCategory } from '../../../../constants/shared-types';

type AccuracyMetric = { label: string; value: number };

type BenchmarkMetricKey = keyof BenchmarkMetrics;

const ACCURACY_METRIC_LABELS: Partial<Record<BenchmarkMetricKey, string>> = {
    imagenet_top1_accuracy: 'Top-1 Acc',
    coco_map_50_95: 'mAP',
    coco_map_50: 'mAP50',
};

export const getAccuracyMetric = (
    modelArchitecture: ModelArchitectureWithPerformanceCategory
): AccuracyMetric | undefined => {
    const benchmarkMetrics = modelArchitecture.stats.benchmark_metrics;

    for (const [key, label] of Object.entries(ACCURACY_METRIC_LABELS)) {
        const value = benchmarkMetrics[key as BenchmarkMetricKey];

        if (!isNil(value)) {
            return { label, value };
        }
    }

    return undefined;
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
