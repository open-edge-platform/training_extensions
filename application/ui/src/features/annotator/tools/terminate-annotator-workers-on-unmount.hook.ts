// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { useQueryClient } from '@tanstack/react-query';

/**
 * Terminate every annotator-tool Worker (SAM, SSIM, magnetic-lasso) and drop
 * the matching tanstack-query entries when the host component unmounts.
 *
 * Mounted at the **dataset section** (`DatasetSection` in `src/router.tsx`)
 * — NOT inside the annotator itself — so the workers survive within-section
 * navigation (dataset list ↔ item ↔ video frame ↔ annotator) and are only
 * freed when the user leaves the dataset section (to Models, Inference, the
 * projects list, etc.). Re-entering the annotator from any sibling URL
 * inside `/projects/:id/dataset/*` is instant.
 *
 * Required because `useSegmentAnythingWorker` sets `gcTime: Infinity` to
 * defeat tanstack's 5-min eviction (which would re-spawn the worker and
 * re-download ~5×ort + ~5×opencv + the multi-MB SAM `.onnx` blobs every
 * time SAM is unmounted). Without an explicit terminator at SOME boundary,
 * the worker (and its ~100 MB WASM heap of model bytes + ORT + OpenCV
 * runtimes) would persist for the lifetime of the tab.
 */
export const useTerminateAnnotatorWorkersOnUnmount = () => {
    const queryClient = useQueryClient();

    useEffect(() => {
        return () => {
            void queryClient.cancelQueries({ queryKey: ['workers'] });

            // Catch already-resolved workers (cancellation is a no-op for
            // settled queries, so they need explicit termination).
            const workerQueries = queryClient.getQueryCache().findAll({ queryKey: ['workers'] });

            for (const query of workerQueries) {
                const data = query.state.data as { worker?: unknown } | undefined;

                if (data?.worker instanceof Worker) {
                    data.worker.terminate();
                }
            }

            queryClient.removeQueries({ queryKey: ['workers'] });
        };
    }, [queryClient]);
};
