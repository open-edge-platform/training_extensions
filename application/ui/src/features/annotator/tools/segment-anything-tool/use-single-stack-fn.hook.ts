// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useRef } from 'react';

export const useSingleStackFn = <
    Callback extends (...args: Parameters<Callback>) => Promise<Awaited<ReturnType<Callback>>>,
>(
    fn: Callback
) => {
    const resolveRef = useRef<() => void>(undefined);
    const rejectRef = useRef<() => void>(undefined);
    const isProcessing = useRef(false);

    const wrappedFn = useCallback(
        async (...args: Parameters<Callback>): Promise<Awaited<ReturnType<Callback>>> => {
            // Wait for the previous function call to be finished
            await new Promise<void>(async (resolve, reject) => {
                // Continue on if we are not waiting for the result of a previous invokation
                if (!isProcessing.current) {
                    return resolve();
                }

                // If the function was invoked while waiting for the previous result then
                // we reject the previous invocation
                if (rejectRef.current) {
                    rejectRef.current();
                    rejectRef.current = undefined;
                    resolveRef.current = undefined;
                }

                // Let the previous invocation resolve this call, or let any subsequent calls
                // cancel this call
                rejectRef.current = reject;
                resolveRef.current = resolve;
            });

            try {
                isProcessing.current = true;
                const result = await fn(...args);
                return result;
            } catch (error) {
                // Reject subsequent invocations as something unexpected made the current invocation fail
                if (rejectRef.current) {
                    rejectRef.current();
                    rejectRef.current = undefined;
                    resolveRef.current = undefined;
                }
                throw error;
            } finally {
                isProcessing.current = false;

                // Resolve any subsequent invocations that were waiting for this function to complete
                if (resolveRef.current) {
                    resolveRef.current();
                    rejectRef.current = undefined;
                    resolveRef.current = undefined;
                }
            }
        },
        [fn]
    );

    return wrappedFn;
};
