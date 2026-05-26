// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getColor } from './util';

describe('getColor', () => {
    it.each([
        [75, 'var(--moss-tint-1)'],
        [90, 'var(--moss-tint-1)'],
        [100, 'var(--moss-tint-1)'],
    ])('returns the green color for accuracy %i (>= 75)', (accuracy, expected) => {
        expect(getColor(accuracy)).toBe(expected);
    });

    it.each([
        [40, 'var(--brand-daisy)'],
        [50, 'var(--brand-daisy)'],
        [74, 'var(--brand-daisy)'],
    ])('returns the yellow color for accuracy %i (40-74)', (accuracy, expected) => {
        expect(getColor(accuracy)).toBe(expected);
    });

    it.each([
        [0, 'var(--coral-shade-1)'],
        [10, 'var(--coral-shade-1)'],
        [39, 'var(--coral-shade-1)'],
    ])('returns the red color for accuracy %i (< 40)', (accuracy, expected) => {
        expect(getColor(accuracy)).toBe(expected);
    });
});
