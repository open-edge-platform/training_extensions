// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderHook } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMultipleMockedMediaItems } from 'mocks/mock-media-item';
import { vi } from 'vitest';

import { useKeyboardNavigation } from './use-keyboard-navigation.hook';

describe('useKeyboardNavigation', () => {
    const items = getMultipleMockedMediaItems(3);

    it('calls onSelectedMediaItem with previous item on ArrowUp', async () => {
        const mockedOnSelectedMediaItem = vi.fn();
        renderHook(() =>
            useKeyboardNavigation({
                items,
                selectedIndex: 1,
                onSelectedMediaItem: mockedOnSelectedMediaItem,
            })
        );

        await userEvent.keyboard('{arrowup}');
        expect(mockedOnSelectedMediaItem).toHaveBeenCalledWith(items[0]);
    });

    it('calls onSelectedMediaItem with next item on ArrowDown', async () => {
        const mockedOnSelectedMediaItem = vi.fn();
        renderHook(() =>
            useKeyboardNavigation({
                items,
                selectedIndex: 1,
                onSelectedMediaItem: mockedOnSelectedMediaItem,
            })
        );

        await userEvent.keyboard('{arrowdown}');
        expect(mockedOnSelectedMediaItem).toHaveBeenCalledWith(items[2]);
    });

    it('does not go below index 0 on ArrowUp', async () => {
        const mockedOnSelectedMediaItem = vi.fn();
        renderHook(() =>
            useKeyboardNavigation({
                items,
                selectedIndex: 0,
                onSelectedMediaItem: mockedOnSelectedMediaItem,
            })
        );

        await userEvent.keyboard('{arrowup}');
        expect(mockedOnSelectedMediaItem).toHaveBeenCalledWith(items[0]);
    });

    it('does not go above last index on ArrowDown', async () => {
        const mockedOnSelectedMediaItem = vi.fn();
        renderHook(() =>
            useKeyboardNavigation({
                items,
                selectedIndex: 2,
                onSelectedMediaItem: mockedOnSelectedMediaItem,
            })
        );

        await userEvent.keyboard('{arrowdown}');
        expect(mockedOnSelectedMediaItem).toHaveBeenCalledWith(items[2]);
    });

    it('does not call onSelectedMediaItem for other keys', async () => {
        const mockedOnSelectedMediaItem = vi.fn();
        renderHook(() =>
            useKeyboardNavigation({
                items,
                selectedIndex: 1,
                onSelectedMediaItem: mockedOnSelectedMediaItem,
            })
        );
        await userEvent.keyboard('{enter}');
        expect(mockedOnSelectedMediaItem).not.toHaveBeenCalled();
    });

    it('does nothing if items is empty', async () => {
        const mockedOnSelectedMediaItem = vi.fn();

        renderHook(() =>
            useKeyboardNavigation({
                items: [],
                selectedIndex: 0,
                onSelectedMediaItem: mockedOnSelectedMediaItem,
            })
        );
        await userEvent.keyboard('{arrowdown}');
        expect(mockedOnSelectedMediaItem).not.toHaveBeenCalled();
    });
});
