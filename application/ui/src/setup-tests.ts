// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import '@testing-library/jest-dom';

import fetchPolyfill, { Request as RequestPolyfill } from 'node-fetch';
import { afterAll, afterEach, beforeAll } from 'vitest';

import { server } from './msw-node-setup';

import './test-utils/mock-event-source';

process.env.PUBLIC_API_BASE_URL = 'http://localhost:7860';

beforeAll(() => {
    server.listen({ onUnhandledRequest: 'bypass' });
});

afterEach(() => {
    server.resetHandlers();
});

afterAll(() => {
    server.close();
});

// Why we need these polyfills:
// https://github.com/reduxjs/redux-toolkit/issues/4966#issuecomment-3115230061
Object.defineProperty(global, 'fetch', {
    // MSW will overwrite this to intercept requests
    writable: true,
    value: fetchPolyfill,
});

Object.defineProperty(global, 'Request', {
    writable: false,
    value: RequestPolyfill,
});

// Mock ResizeObserver which is not available in jsdom
class ResizeObserverMock {
    observe = vi.fn();
    unobserve = vi.fn();
    disconnect = vi.fn();
}

global.ResizeObserver = ResizeObserverMock;
