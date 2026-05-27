// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { TrainingDevice } from '../../../../constants/shared-types';
import { createDeviceName, getDefaultTrainingDevice } from './utils';

const makeDevice = (overrides: Partial<TrainingDevice> = {}): TrainingDevice => ({
    type: 'cpu',
    name: 'CPU',
    memory: null,
    index: null,
    ...overrides,
});

describe('getDefaultTrainingDevice', () => {
    it('returns undefined when no devices are available', () => {
        expect(getDefaultTrainingDevice([])).toBeUndefined();
    });

    it('returns the only device when there is just one (CPU)', () => {
        const cpu = makeDevice();
        expect(getDefaultTrainingDevice([cpu])).toBe(cpu);
    });

    it('prefers xpu over cpu', () => {
        const cpu = makeDevice({ type: 'cpu', name: 'CPU' });
        const xpu = makeDevice({ type: 'xpu', name: 'Intel Arc A770', memory: 8_000_000_000, index: 0 });

        expect(getDefaultTrainingDevice([cpu, xpu])).toBe(xpu);
    });

    it('prefers cuda over cpu', () => {
        const cpu = makeDevice({ type: 'cpu', name: 'CPU' });
        const cuda = makeDevice({ type: 'cuda', name: 'NVIDIA RTX 4090', memory: 25_769_803_776, index: 0 });

        expect(getDefaultTrainingDevice([cpu, cuda])).toBe(cuda);
    });

    it('picks the xpu with the highest memory when multiple xpu devices are available', () => {
        const cpu = makeDevice({ type: 'cpu', name: 'CPU' });
        const xpuLow = makeDevice({ type: 'xpu', name: 'Intel iGPU', memory: 8_589_934_592, index: 1 });
        const xpuHigh = makeDevice({ type: 'xpu', name: 'Intel Arc A770', memory: 25_769_803_776, index: 0 });

        expect(getDefaultTrainingDevice([cpu, xpuHigh, xpuLow])).toBe(xpuHigh);
    });

    it('picks the device with the lowest index when memory is the same', () => {
        const cpu = makeDevice({ type: 'cpu', name: 'CPU' });
        const xpu0 = makeDevice({ type: 'xpu', name: 'Intel Arc 0', memory: 8_000_000_000, index: 0 });
        const xpu1 = makeDevice({ type: 'xpu', name: 'Intel Arc 1', memory: 8_000_000_000, index: 1 });

        expect(getDefaultTrainingDevice([cpu, xpu1, xpu0])).toBe(xpu0);
    });

    it('picks the xpu/cuda device with the highest memory across mixed gpu types', () => {
        const cpu = makeDevice({ type: 'cpu', name: 'CPU' });
        const xpu = makeDevice({ type: 'xpu', name: 'Intel Arc A770', memory: 25_769_803_776, index: 0 });
        const cuda = makeDevice({ type: 'cuda', name: 'NVIDIA RTX 3080', memory: 10_737_418_240, index: 0 });

        expect(getDefaultTrainingDevice([cpu, xpu, cuda])).toBe(xpu);
    });

    it('falls back to the first device (CPU) when no gpu devices are present', () => {
        const cpu = makeDevice({ type: 'cpu', name: 'CPU' });

        expect(getDefaultTrainingDevice([cpu])).toBe(cpu);
    });

    it('uses the example from the issue: picks Arc A770 over iGPU', () => {
        const cpu = makeDevice({ type: 'cpu', name: 'CPU', memory: null, index: null });
        const arcA770 = makeDevice({ type: 'xpu', name: 'Intel Arc A770', memory: 25_769_803_776, index: 0 });
        const iGPU = makeDevice({ type: 'xpu', name: 'Intel iGPU', memory: 8_589_934_592, index: 1 });

        expect(getDefaultTrainingDevice([cpu, arcA770, iGPU])).toBe(arcA770);
    });
});

describe('createDeviceName', () => {
    it('returns just the name when memory and index are both null', () => {
        expect(createDeviceName(makeDevice({ name: 'CPU', memory: null, index: null }))).toBe('CPU');
    });

    it('returns just the name when memory and index are undefined', () => {
        expect(createDeviceName(makeDevice({ name: 'CPU' }))).toBe('CPU');
    });

    it('appends memory in GB (rounded up) when memory is set', () => {
        expect(createDeviceName(makeDevice({ name: 'GPU', memory: 8_000_000_000 }))).toBe('GPU (8 GB)');
    });

    it('rounds up fractional GB values with Math.ceil', () => {
        // 1 byte over 7 GB → rounds up to 8 GB
        expect(createDeviceName(makeDevice({ name: 'GPU', memory: 7 * 1024 ** 3 + 1 }))).toBe('GPU (8 GB)');
    });

    it('appends index in brackets when index is set', () => {
        expect(createDeviceName(makeDevice({ name: 'XPU', index: 0 }))).toBe('XPU [0]');
    });

    it('appends memory before index when both are set', () => {
        expect(createDeviceName(makeDevice({ name: 'XPU', memory: 21_359_386_624, index: 1 }))).toBe('XPU (20 GB) [1]');
    });

    it('appends memory when memory is 0 (0 != null is true)', () => {
        expect(createDeviceName(makeDevice({ name: 'CPU', memory: 0 }))).toBe('CPU (0 GB)');
    });
});
