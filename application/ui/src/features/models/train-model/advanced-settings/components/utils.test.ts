// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getStep } from './utils';

describe('getStep', () => {
    it('returns the provided step value for float type', () => {
        expect(getStep({ step: 0.5, minValue: 0.0, maxValue: 1.0, type: 'float' })).toBe(0.5);
    });

    describe('when step is undefined and type is int', () => {
        it('returns the default int step of 1', () => {
            expect(getStep({ minValue: 0, maxValue: 100, type: 'int' })).toBe(1);
        });

        it('returns the default int step of 1 when minValue and maxValue are null', () => {
            expect(getStep({ minValue: null, maxValue: null, type: 'int' })).toBe(1);
        });

        it('returns the default int step of 1 when only minValue is null', () => {
            expect(getStep({ minValue: null, maxValue: 100, type: 'int' })).toBe(1);
        });

        it('returns the default int step of 1 when only maxValue is null', () => {
            expect(getStep({ minValue: 0, maxValue: null, type: 'int' })).toBe(1);
        });
    });

    describe('when step is undefined, type is float, and minValue or maxValue is null', () => {
        it('returns the default float step of 0.1 when maxValue is null', () => {
            expect(getStep({ minValue: 0.0, maxValue: null, type: 'float' })).toBe(0.1);
        });

        it('returns the default float step of 0.1 when minValue is null', () => {
            expect(getStep({ minValue: null, maxValue: 1.0, type: 'float' })).toBe(0.1);
        });

        it('returns the default float step of 0.1 when both are null', () => {
            expect(getStep({ minValue: null, maxValue: null, type: 'float' })).toBe(0.1);
        });
    });

    describe('when step is undefined, type is float, and both minValue and maxValue are provided', () => {
        it('computes a fine-grained step for a range of 1 (0.0 to 1.0)', () => {
            // range = 1.0 - 0.0 = 1, log10(1) = 0, ceil(|0|) = 0, but || 1 yields exponent = 1
            // step = 1 / 10^(1+3) = 1 / 10000 = 0.0001
            expect(getStep({ minValue: 0.0, maxValue: 1.0, type: 'float' })).toBeCloseTo(0.0001);
        });

        it('computes a fine-grained step for a range of 0.1 (0.0 to 0.1)', () => {
            // range = 0.1, log10(0.1) = -1, ceil(|-1|) = 1, exponent = 1
            // step = 1 / 10^(1+3) = 1 / 10000 = 0.0001
            expect(getStep({ minValue: 0.0, maxValue: 0.1, type: 'float' })).toBeCloseTo(0.0001);
        });

        it('computes a fine-grained step for a negative minValue range (-1.0 to 1.0)', () => {
            // range = 1.0 - (-1.0) = 2, log10(2) ≈ 0.301, ceil(|0.301|) = 1, exponent = 1
            // step = 1 / 10^(1+3) = 1 / 10000 = 0.0001
            expect(getStep({ minValue: -1.0, maxValue: 1.0, type: 'float' })).toBeCloseTo(0.0001);
        });

        it('computes a fine-grained step when min equals max (range of 0)', () => {
            // range = 5.0 - 5.0 = 0, log10(0) = -Infinity, ceil(|-Infinity|) = Infinity
            // 1 / 10^(Infinity+3) = 0
            expect(getStep({ minValue: 5.0, maxValue: 5.0, type: 'float' })).toBe(0);
        });
    });
});
