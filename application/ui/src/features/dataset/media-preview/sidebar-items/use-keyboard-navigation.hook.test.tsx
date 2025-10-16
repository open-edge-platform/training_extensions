// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef } from 'react';

import { fireEvent } from '@testing-library/react';
import { getMultipleMockedMediaItems } from 'mocks/mock-media-item';
import { render, screen } from 'test-utils/render';
import { vi } from 'vitest';

import { useKeyboardNavigation, UseKeyboardNavigationProps } from './use-keyboard-navigation.hook';

const App = ({ ...options }: Omit<UseKeyboardNavigationProps, 'ref'>) => {
    const ref = useRef(null);
    useKeyboardNavigation({ ...options, ref });

    return <div ref={ref} data-testid='target' />;
};

describe('useKeyboardNavigation', () => {
    const items = getMultipleMockedMediaItems(3);

    it('calls onSelectedMediaItem with previous item on ArrowUp', async () => {
        const mockedOnSelectedMediaItem = vi.fn();
        render(<App items={items} onSelectedMediaItem={mockedOnSelectedMediaItem} selectedIndex={0} />);

        fireEvent.keyDown(screen.getByTestId('target'), { key: 'ArrowUp' });
        expect(mockedOnSelectedMediaItem).toHaveBeenCalledWith(items[0]);
    });

    it('calls onSelectedMediaItem with next item on ArrowDown', async () => {
        const mockedOnSelectedMediaItem = vi.fn();
        render(<App items={items} onSelectedMediaItem={mockedOnSelectedMediaItem} selectedIndex={1} />);

        fireEvent.keyDown(screen.getByTestId('target'), { key: 'ArrowDown' });
        expect(mockedOnSelectedMediaItem).toHaveBeenCalledWith(items[2]);
    });

    it('does not go below index 0 on ArrowUp', async () => {
        const mockedOnSelectedMediaItem = vi.fn();
        render(<App items={items} onSelectedMediaItem={mockedOnSelectedMediaItem} selectedIndex={0} />);

        fireEvent.keyDown(screen.getByTestId('target'), { key: 'ArrowUp' });
        expect(mockedOnSelectedMediaItem).toHaveBeenCalledWith(items[0]);
    });

    it('does not go above last index on ArrowDown', async () => {
        const mockedOnSelectedMediaItem = vi.fn();
        render(<App items={items} onSelectedMediaItem={mockedOnSelectedMediaItem} selectedIndex={2} />);

        fireEvent.keyDown(screen.getByTestId('target'), { key: 'ArrowDown' });
        expect(mockedOnSelectedMediaItem).toHaveBeenCalledWith(items[2]);
    });

    it('does not call onSelectedMediaItem for other keys', async () => {
        const mockedOnSelectedMediaItem = vi.fn();
        render(<App items={items} onSelectedMediaItem={mockedOnSelectedMediaItem} selectedIndex={1} />);

        fireEvent.keyDown(screen.getByTestId('target'), { key: 'Enter' });
        expect(mockedOnSelectedMediaItem).not.toHaveBeenCalled();
    });

    it('does nothing if items is empty', async () => {
        const mockedOnSelectedMediaItem = vi.fn();
        render(<App items={[]} onSelectedMediaItem={mockedOnSelectedMediaItem} selectedIndex={0} />);

        fireEvent.keyDown(screen.getByTestId('target'), { key: 'ArrowDown' });
        expect(mockedOnSelectedMediaItem).not.toHaveBeenCalled();
    });
});
