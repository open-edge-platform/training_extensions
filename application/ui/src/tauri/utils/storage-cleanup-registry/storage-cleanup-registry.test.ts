// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { waitFor } from '@testing-library/react';

import { setupTauriStorageCleanup } from './storage-cleanup-registry';

// Restore the real jsdom localStorage so Object.keys(localStorage) returns stored keys
vi.unstubAllGlobals();

const { mockOnCloseRequested, mockGetCurrentWindow } = vi.hoisted(() => {
    const onCloseRequested = vi.fn();
    const getCurrentWindow = vi.fn(() => ({ onCloseRequested }));

    return { mockOnCloseRequested: onCloseRequested, mockGetCurrentWindow: getCurrentWindow };
});

vi.mock('@tauri-apps/api/window', () => ({
    getCurrentWindow: mockGetCurrentWindow,
}));

const TAURI_KEY = '__TAURI_INTERNALS__';
const enableTauri = () => Object.defineProperty(window, TAURI_KEY, { value: {}, configurable: true });
const disableTauri = () => Reflect.deleteProperty(window, TAURI_KEY);

describe('setupTauriStorageCleanup', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        localStorage.clear();
        disableTauri();
    });

    it('does nothing when not running in Tauri', () => {
        setupTauriStorageCleanup();

        expect(mockGetCurrentWindow).not.toHaveBeenCalled();
    });

    it('registers a close handler in Tauri environment', async () => {
        enableTauri();

        setupTauriStorageCleanup();

        await waitFor(() => {
            expect(mockGetCurrentWindow).toHaveBeenCalled();
            expect(mockOnCloseRequested).toHaveBeenCalledWith(expect.any(Function));
        });
    });

    it('removes dataset storage keys on close', async () => {
        enableTauri();

        localStorage.setItem('export-dataset-123', 'data');
        localStorage.setItem('export-dataset-456', 'data');
        localStorage.setItem('import-dataset-as-new-project', 'data');
        localStorage.setItem('import-dataset-to-project-abc', 'data');
        localStorage.setItem('unrelated-key', 'keep');

        setupTauriStorageCleanup();

        await waitFor(() => {
            const cleanupHandler = mockOnCloseRequested.mock.calls[0][0];
            cleanupHandler();
        });

        expect(localStorage.getItem('export-dataset-123')).toBeNull();
        expect(localStorage.getItem('export-dataset-456')).toBeNull();
        expect(localStorage.getItem('import-dataset-as-new-project')).toBeNull();
        expect(localStorage.getItem('import-dataset-to-project-abc')).toBeNull();
        expect(localStorage.getItem('unrelated-key')).toBe('keep');
    });

    it('preserves non-dataset keys on close', async () => {
        enableTauri();

        localStorage.setItem('user-settings', 'value');
        localStorage.setItem('auth-token', 'value');

        setupTauriStorageCleanup();

        await waitFor(() => {
            const cleanupHandler = mockOnCloseRequested.mock.calls[0][0];
            cleanupHandler();
        });

        expect(localStorage.getItem('user-settings')).toBe('value');
        expect(localStorage.getItem('auth-token')).toBe('value');
    });
});
