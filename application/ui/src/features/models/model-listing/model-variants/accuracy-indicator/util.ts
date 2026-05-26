// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export const getColor = (accuracy: number): string => {
    if (accuracy >= 75) return 'var(--moss-tint-1)';
    if (accuracy >= 40 && accuracy <= 74) return 'var(--brand-daisy)';

    return 'var(--coral-shade-1)';
};
