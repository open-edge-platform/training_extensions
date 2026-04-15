// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { formatCompactDuration } from './util';

describe('formatCompactDuration', () => {
    it('formats seconds only as mm:ss', () => {
        expect(formatCompactDuration(45)).toBe('00:45');
    });

    it('formats minutes and seconds as mm:ss', () => {
        expect(formatCompactDuration(125)).toBe('02:05');
    });

    it('formats exactly one hour as HH:mm:ss', () => {
        expect(formatCompactDuration(3600)).toBe('01:00:00');
    });

    it('formats hours, minutes, and seconds as HH:mm:ss', () => {
        expect(formatCompactDuration(3661)).toBe('01:01:01');
    });

    it('formats zero seconds as mm:ss', () => {
        expect(formatCompactDuration(0)).toBe('00:00');
    });

    it('formats 59 minutes 59 seconds without hours as mm:ss', () => {
        expect(formatCompactDuration(3599)).toBe('59:59');
    });

    it('formats multiple hours as HH:mm:ss', () => {
        expect(formatCompactDuration(7384)).toBe('02:03:04');
    });
});
