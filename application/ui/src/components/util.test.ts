// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createDeviceName } from './util';

describe('createDeviceName', () => {
    it('returns just the name when memory and index are both null', () => {
        expect(createDeviceName({ name: 'CPU', memory: null, index: null })).toBe('CPU');
    });

    it('returns just the name when memory and index are undefined', () => {
        expect(createDeviceName({ name: 'CPU' })).toBe('CPU');
    });

    it('appends memory in GB (rounded up) when memory is set', () => {
        expect(createDeviceName({ name: 'GPU', memory: 8_000_000_000 })).toBe('GPU (8 GB)');
    });

    it('rounds up fractional GB values with Math.ceil', () => {
        // 1 byte over 7 GB → rounds up to 8 GB
        expect(createDeviceName({ name: 'GPU', memory: 7 * 1024 ** 3 + 1 })).toBe('GPU (8 GB)');
    });

    it('appends index in brackets when index is set', () => {
        expect(createDeviceName({ name: 'XPU', index: 0 })).toBe('XPU [0]');
    });

    it('appends memory before index when both are set', () => {
        expect(createDeviceName({ name: 'XPU', memory: 21_359_386_624, index: 1 })).toBe('XPU (20 GB) [1]');
    });

    it('appends memory when memory is 0 (0 != null is true)', () => {
        expect(createDeviceName({ name: 'CPU', memory: 0 })).toBe('CPU (0 GB)');
    });
});
