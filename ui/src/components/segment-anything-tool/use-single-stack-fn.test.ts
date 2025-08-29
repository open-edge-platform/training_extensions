// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, renderHook, waitFor } from '@testing-library/react';

import { useSingleStackFn } from './use-single-stack-fn.hook';

describe('useSingleStackFn', () => {
    test('it runs the function', async () => {
        const fn = vi.fn();
        const { result } = renderHook(() => useSingleStackFn(fn));

        await act(async () => {
            await result.current(1);
        });

        await waitFor(() => {
            expect(fn).toHaveBeenCalledWith(1);
        });

        await act(async () => {
            await result.current(2);
        });

        await waitFor(() => {
            expect(fn).toHaveBeenCalledWith(2);
        });
    });

    it('skips consecutive calls', async () => {
        const resolve = vi.fn();
        const reject = vi.fn();

        // We setup a function fn, that waits until our test code triggers its
        // resolve function by calling resolveFn[n]()
        const resolveFn: Record<number, (value: unknown) => void> = {};
        const fn = async (n: number) => {
            return new Promise((resolvePromise) => {
                resolveFn[n] = resolvePromise;
            });
        };

        const { result } = renderHook(() => useSingleStackFn(fn));

        act(() => {
            result
                .current(3)
                .then(() => resolve(3))
                .catch(() => reject(3));
        });

        // Trigger hook
        await waitFor(() => {
            expect(resolveFn[3]).toBeDefined();
        });

        act(() => {
            result
                .current(4)
                .then(() => resolve(4))
                .catch(() => reject(4));
        });

        act(() => {
            result
                .current(5)
                .then(() => resolve(5))
                .catch(() => reject(5));
        });

        act(() => {
            resolveFn[3]('test');
        });

        await waitFor(() => {
            expect(resolve).toHaveBeenCalledWith(3);
        });

        act(() => {
            resolveFn[5]('test');
        });

        await waitFor(() => {
            expect(resolve).toHaveBeenCalledWith(5);
        });

        // One final verify that 3 and 5 have been resolved while the
        // intermediate call, 4, has been rejected
        expect(resolve).toHaveBeenCalledWith(3);
        expect(resolve).not.toHaveBeenCalledWith(4);
        expect(resolve).toHaveBeenCalledWith(5);

        expect(reject).not.toHaveBeenCalledWith(3);
        expect(reject).toHaveBeenCalledWith(4);
        expect(reject).not.toHaveBeenCalledWith(5);
    });
});
