// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { connectSSE } from './fetch-sse';

type MockEventSource = {
    onopen: ((event: Event) => void) | null;
    onmessage: ((event: MessageEvent) => void) | null;
    onerror: ((event: Event) => void) | null;
    close: ReturnType<typeof vi.fn>;
    url: string;
    readyState: number;
};

let mockEventSourceInstances: MockEventSource[] = [];

const MockEventSourceConstructor = vi.fn().mockImplementation((url: string) => {
    const instance: MockEventSource = {
        onopen: null,
        onmessage: null,
        onerror: null,
        close: vi.fn(),
        url,
        readyState: 0,
    };
    mockEventSourceInstances.push(instance);
    return instance;
});

vi.stubGlobal('EventSource', MockEventSourceConstructor);

const getLastEventSource = (): MockEventSource => {
    const instance = mockEventSourceInstances.at(-1);
    if (!instance) {
        throw new Error('No MockEventSource instance found');
    }
    return instance;
};

const simulateMessage = (instance: MockEventSource, data: string) => {
    instance.onmessage?.({ data } as MessageEvent);
};

const simulateOpen = (instance: MockEventSource) => {
    instance.onopen?.({} as Event);
};

const simulateError = (instance: MockEventSource) => {
    instance.onerror?.({} as Event);
};

afterAll(() => {
    vi.unstubAllGlobals();
    mockEventSourceInstances = [];
});

describe('connectSSE', () => {
    beforeEach(() => {
        mockEventSourceInstances = [];
        MockEventSourceConstructor.mockClear();
    });

    it('creates an EventSource with the correct URL including API base', () => {
        connectSSE('/api/jobs/123/status', { onMessage: vi.fn() });

        expect(MockEventSourceConstructor).toHaveBeenCalledWith(expect.stringContaining('/api/jobs/123/status'));
    });

    it('parses and delivers JSON messages via onMessage callback', () => {
        const onMessage = vi.fn();
        connectSSE<{ value: number }>('/api/test', { onMessage });
        const eventSource = getLastEventSource();

        simulateMessage(eventSource, JSON.stringify({ value: 42 }));

        expect(onMessage).toHaveBeenCalledWith({ value: 42 });
    });

    it('delivers multiple messages in order', () => {
        const messages: number[] = [];
        connectSSE<{ n: number }>('/api/test', {
            onMessage: (data) => messages.push(data.n),
        });
        const eventSource = getLastEventSource();

        simulateMessage(eventSource, JSON.stringify({ n: 1 }));
        simulateMessage(eventSource, JSON.stringify({ n: 2 }));
        simulateMessage(eventSource, JSON.stringify({ n: 3 }));
        expect(messages).toEqual([1, 2, 3]);
    });

    it('closes the connection and resolves done when the server closes the connection', async () => {
        const onMessage = vi.fn();
        const onClose = vi.fn();
        const { done } = connectSSE('/api/test', { onMessage, onClose });
        const eventSource = getLastEventSource();
        simulateError(eventSource);

        await done;

        expect(onClose).toHaveBeenCalledOnce();
        expect(eventSource.close).toHaveBeenCalledOnce();
    });

    it('calls onOpen when the connection is established', () => {
        const onOpen = vi.fn();
        connectSSE('/api/test', { onMessage: vi.fn(), onOpen });
        const eventSource = getLastEventSource();

        simulateOpen(eventSource);
        expect(onOpen).toHaveBeenCalledOnce();
    });

    it('calls onError and onClose when an error occurs, then resolves done', async () => {
        const onError = vi.fn();
        const onClose = vi.fn();
        const { done } = connectSSE('/api/test', { onMessage: vi.fn(), onError, onClose });
        const eventSource = getLastEventSource();

        simulateError(eventSource);

        await done;

        expect(onError).toHaveBeenCalledOnce();
        expect(onClose).toHaveBeenCalledOnce();
        expect(eventSource.close).toHaveBeenCalledOnce();
    });

    it('skips unparseable messages without throwing', () => {
        const onMessage = vi.fn();
        connectSSE('/api/test', { onMessage });
        const eventSource = getLastEventSource();

        simulateMessage(eventSource, 'not valid json');
        expect(onMessage).not.toHaveBeenCalled();
    });

    it('close() is idempotent — calling multiple times only triggers onClose once', () => {
        const onClose = vi.fn();
        const { close } = connectSSE('/api/test', { onMessage: vi.fn(), onClose });
        const eventSource = getLastEventSource();

        close();
        close();
        close();

        expect(onClose).toHaveBeenCalledOnce();
        expect(eventSource.close).toHaveBeenCalledOnce();
    });

    it('manual close() closes the EventSource', () => {
        const { close } = connectSSE('/api/test', { onMessage: vi.fn() });
        const eventSource = getLastEventSource();

        simulateOpen(eventSource);
        close();

        expect(eventSource.close).toHaveBeenCalledOnce();
    });

    it('returns the underlying EventSource instance', () => {
        const { eventSource } = connectSSE('/api/test', { onMessage: vi.fn() });
        const lastEventSource = getLastEventSource();

        expect(eventSource).toBe(lastEventSource);
    });
});
