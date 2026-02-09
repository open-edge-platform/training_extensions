// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type Model } from '../../../../../constants/shared-types';

export const getTestingMetric = (model: Model): { name: string; value: number } | undefined => {
    const evaluation = model.evaluations.find(({ subset }) => subset === 'testing');
    const primaryMetric = evaluation?.metrics.find(({ primary }) => primary);

    if (primaryMetric !== undefined) {
        return { name: primaryMetric.name, value: Math.round(primaryMetric.value * 100) };
    }
};

export const getFirstAvailableTestingMetric = (
    models: Model[] | undefined
): { name: string; value: number } | undefined => {
    // Should never happen, but just in case
    if (models === undefined) {
        return undefined;
    }

    for (const model of models) {
        const testingMetric = getTestingMetric(model);

        if (testingMetric !== undefined) {
            return testingMetric;
        }
    }

    return undefined;
};
