// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { formatDurationText } from './time-utils';

describe('formatDurationText', () => {
    it('formats zero duration as 00:00:00', () => {
        expect(formatDurationText(0)).toBe('00:00:00');
    });

    it('formats seconds only', () => {
        expect(formatDurationText(5)).toBe('00:00:05');
        expect(formatDurationText(59)).toBe('00:00:59');
    });

    it('formats minutes and seconds', () => {
        expect(formatDurationText(60)).toBe('00:01:00');
        expect(formatDurationText(90)).toBe('00:01:30');
        expect(formatDurationText(3599)).toBe('00:59:59');
    });

    it('formats hours, minutes, and seconds', () => {
        expect(formatDurationText(3600)).toBe('01:00:00');
        expect(formatDurationText(3661)).toBe('01:01:01');
        expect(formatDurationText(7384)).toBe('02:03:04');
    });

    it('pads single-digit values with leading zeros', () => {
        expect(formatDurationText(3661)).toBe('01:01:01');
    });

    it('handles large durations beyond 99 hours', () => {
        expect(formatDurationText(360000)).toBe('100:00:00');
    });

    it('floors fractional seconds', () => {
        expect(formatDurationText(1.9)).toBe('00:00:01');
        expect(formatDurationText(59.999)).toBe('00:00:59');
        expect(formatDurationText(3600.7)).toBe('01:00:00');
    });
});
