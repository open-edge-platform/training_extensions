// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef } from 'react';

import { connectSSE, type SSEOptions } from '../api/fetch-sse';

type SSEConnection = {
    close: () => void;
};

export const useSSE = <T>(url: string | undefined, options: SSEOptions<T>): SSEConnection => {
    const optionsRef = useRef(options);
    optionsRef.current = options;

    const connectionRef = useRef<ReturnType<typeof connectSSE<T>> | null>(null);

    useEffect(() => {
        if (!url) {
            return;
        }

        const connection = connectSSE<T>(url, {
            onMessage: (data) => optionsRef.current.onMessage(data),
            onError: (error) => optionsRef.current.onError?.(error),
            onOpen: () => optionsRef.current.onOpen?.(),
            onClose: () => optionsRef.current.onClose?.(),
            retry: optionsRef.current.retry,
        });

        connectionRef.current = connection;

        return () => {
            connection.close();
            connectionRef.current = null;
        };
    }, [url]);

    const close = useCallback(() => {
        connectionRef.current?.close();
    }, []);

    return { close };
};
