// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { API_BASE_URL } from './client';

type SSEConnection = {
    close: () => void;
    done: Promise<void>;
    eventSource: EventSource;
};

export type SSEOptions<T> = {
    onMessage: (data: T) => void;
    onError?: (error: Event) => void;
    onOpen?: () => void;
    onClose?: () => void;
    onRetry?: () => void;
    retry?: boolean;
};

const INITIAL_RETRY_DELAY_MS = 1_000;
const MAX_RETRIES = 7;

export const connectSSE = <T = unknown>(path: string, options: SSEOptions<T>): SSEConnection => {
    const url = `${API_BASE_URL}${path}`;

    let closed = false;
    let retryDelay = INITIAL_RETRY_DELAY_MS;
    let retryCount = 0;
    let retryTimeout: ReturnType<typeof setTimeout> | null = null;
    let resolveDone: (() => void) | null = null;
    let activeSource: EventSource | null = null;

    const createEventSource = (): EventSource => {
        const eventSource = new EventSource(url);
        activeSource = eventSource;

        eventSource.onopen = () => {
            retryDelay = INITIAL_RETRY_DELAY_MS;
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
            if (!closed && options.retry && retryCount < MAX_RETRIES) {
                eventSource.close();
                activeSource = null;

                options.onRetry?.();

                retryTimeout = setTimeout(() => {
                    if (!closed) {
                        createEventSource();
                    }
                }, retryDelay);

                retryDelay = retryDelay * 2;
                retryCount += 1;
            } else {
                options.onError?.(event);
                close();
            }
        };

        return eventSource;
    };

    const close = () => {
        if (!closed) {
            closed = true;

            if (retryTimeout !== null) {
                clearTimeout(retryTimeout);
            }

            activeSource?.close();
            activeSource = null;
            options.onClose?.();
            resolveDone?.();
        }
    };

    const eventSource = createEventSource();

    const done = new Promise<void>((resolve) => {
        resolveDone = resolve;
    });

    return { close, done, eventSource };
};
