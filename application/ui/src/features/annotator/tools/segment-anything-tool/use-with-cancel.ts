// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef } from 'react';

import type { Shape } from '../../../../shared/types';
import { InteractiveAnnotationPoint } from './segment-anything.interface';

export const useWithCancel = (fn: (points: InteractiveAnnotationPoint[]) => Promise<Shape[]>) => {
    const abortController = useRef<AbortController | null>(null);

    const cancellableCallback = useCallback(
        async (...args: Parameters<typeof fn>) => {
            // Cancel any ongoing request
            abortController.current?.abort();

            abortController.current = new AbortController();

            const result = await fn(...args);

            if (abortController.current.signal.aborted) {
                throw new DOMException('Request aborted', 'AbortError');
            }

            return result;
        },
        [fn]
    );

    const cancel = useCallback(() => {
        abortController.current?.abort();
        abortController.current = null;
    }, [abortController]);

    useEffect(() => {
        return () => {
            cancel();
        };
    }, [cancel]);

    return {
        call: cancellableCallback,
        cancel,
    };
};
