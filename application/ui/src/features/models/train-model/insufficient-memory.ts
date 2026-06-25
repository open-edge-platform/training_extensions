// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export interface RecommendedModel {
    id: string;
    name: string;
    estimated_memory_mb: number;
}

export interface InsufficientMemoryDetail {
    code: 'insufficient_memory';
    message: string;
    model_architecture_id: string;
    model_architecture_name: string;
    device: string;
    estimated_memory_mb: number;
    available_memory_mb: number;
    usable_memory_mb: number;
    recommended_models: RecommendedModel[];
}

const isObject = (value: unknown): value is Record<string, unknown> =>
    typeof value === 'object' && value !== null;

/**
 * Extracts the structured "insufficient memory" payload from a failed train-model request, or
 * returns `null` when the error is unrelated.
 */
export const getInsufficientMemoryDetail = (error: unknown): InsufficientMemoryDetail | null => {
    if (!isObject(error) || !isObject(error.detail)) {
        return null;
    }

    const detail = error.detail;
    if (detail.code !== 'insufficient_memory') {
        return null;
    }

    return detail as unknown as InsufficientMemoryDetail;
};

export const isInsufficientMemoryError = (error: unknown): boolean => getInsufficientMemoryDetail(error) !== null;
