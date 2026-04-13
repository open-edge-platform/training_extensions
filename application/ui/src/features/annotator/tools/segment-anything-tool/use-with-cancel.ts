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

            // Create a new controller for THIS call and capture its signal in a local
            // variable BEFORE awaiting anything. Any future cancel() call will replace
            // abortController.current, but `controller.signal` here is a closed-over reference to
            // the abortController for THIS specific invocation and will correctly reflect
            // whether it was aborted — even after the ref has been replaced.
            const controller = new AbortController();
            abortController.current = controller;

            const result = await fn(...args);

            if (controller.signal.aborted) {
                throw new DOMException('Request aborted', 'AbortError');
            }

            return result;
        },
        [fn]
    );

    const cancel = useCallback(() => {
        abortController.current?.abort();
        abortController.current = null;
    }, []);

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
