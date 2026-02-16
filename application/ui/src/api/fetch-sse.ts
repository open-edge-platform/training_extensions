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
    retry?: boolean;
};

const MAX_RETRY_DELAY_MS = 20_000;
const INITIAL_RETRY_DELAY_MS = 1_000;

export const connectSSE = <T = unknown>(path: string, options: SSEOptions<T>): SSEConnection => {
    const url = `${API_BASE_URL}${path}`;

    let closed = false;
    let retryDelay = INITIAL_RETRY_DELAY_MS;
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
            options.onError?.(event);

            if (!closed && options.retry) {
                eventSource.close();
                activeSource = null;

                retryTimeout = setTimeout(() => {
                    if (!closed) {
                        createEventSource();
                    }
                }, retryDelay);

                retryDelay = Math.min(retryDelay * 2, MAX_RETRY_DELAY_MS);
            } else {
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
