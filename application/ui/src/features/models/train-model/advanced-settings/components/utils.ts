// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { NumberConfigurableParameter } from '../../../../../constants/shared-types';

const DEFAULT_INT_STEP = 1;
const DEFAULT_FLOAT_STEP = 0.1;

const getDecimalPoints = (value: number): number => {
    // When log10 returns 0 (log10(1) = 0) we need to return 1
    return Math.abs(Math.ceil(Math.log10(value))) || 1;
};

const getFloatingPointStep = (minValue: number, maxValue: number): number => {
    const exponent = getDecimalPoints(maxValue - minValue);

    return 1 / Math.pow(10, exponent + 3);
};

export const getStep = ({
    step,
    maxValue,
    minValue,
    type,
}: {
    step?: number;
    minValue: number | null;
    maxValue: number | null;
    type: NumberConfigurableParameter['value_type'];
}): number => {
    if (step !== undefined) {
        return step;
    }

    if (type === 'int') {
        return DEFAULT_INT_STEP;
    }

    if (maxValue === null || minValue === null) {
        return DEFAULT_FLOAT_STEP;
    }

    return getFloatingPointStep(minValue, maxValue);
};
