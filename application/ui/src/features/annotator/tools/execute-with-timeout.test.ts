// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { executeWithTimeout } from './execute-with-timeout';

describe('executeWithTimeout', () => {
    beforeEach(() => {
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('returns resolved value before timeout', async () => {
        const promise = executeWithTimeout(Promise.resolve('ok'), 'SAM encoder', 1000);

        await expect(promise).resolves.toBe('ok');
    });

    it('rejects when operation exceeds timeout', async () => {
        const neverResolvingPromise = new Promise<string>(() => undefined);
        const promise = executeWithTimeout(neverResolvingPromise, 'SAM encoder', 10);
        const rejectionExpectation = expect(promise).rejects.toThrow(/SAM encoder timed out after 10ms/i);

        await vi.advanceTimersByTimeAsync(11);

        await rejectionExpectation;
    });
});
