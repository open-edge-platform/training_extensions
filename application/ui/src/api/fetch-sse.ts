// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { API_BASE_URL } from './client';

export type SSEConnection = {
    close: () => void;
    done: Promise<void>;
    eventSource: EventSource;
};

export type SSEOptions<T> = {
    onMessage: (data: T) => void;
    onError?: (error: Event) => void;
    onOpen?: () => void;
    onClose?: () => void;
};

export const connectSSE = <T = unknown>(path: string, options: SSEOptions<T>): SSEConnection => {
    const url = `${API_BASE_URL}${path}`;
    const eventSource = new EventSource(url);

    let closed = false;

    const close = () => {
        if (!closed) {
            closed = true;
            eventSource.close();
            options.onClose?.();
        }
    };

    const done = new Promise<void>((resolve) => {
        eventSource.onopen = () => {
            options.onOpen?.();
        };

        eventSource.onmessage = (event: MessageEvent) => {
            try {
                const data = JSON.parse(event.data) as T;
                options.onMessage(data);
            } catch {
                // Skip unparseable messages
            }
        };

        eventSource.onerror = (event: Event) => {
            options.onError?.(event);
            close();
            resolve();
        };
    });

    return { close, done, eventSource };
};
