// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@testing-library/react';
import { getMultipleMockedMediaItems } from 'mocks/mock-media-item';
import { Size } from 'react-aria-components';
import { MediaState } from 'src/routes/dataset/provider';
import { describe, expect, it } from 'vitest';

import { VirtualizerGridLayout } from './virtualizer-grid-layout.component';

// required configuration; otherwise, the list renders empty
const mockedLayoutOptions = {
    maxColumns: 1,
    maxItemSize: new Size(100, 100),
    maxHorizontalSpace: 1000,
};

describe('VirtualizerGridLayout', () => {
    const mockedItems = getMultipleMockedMediaItems(2, 'test');

    it('renders all items as visible options', () => {
        const mockedLoadingMore = vi.fn();

        render(
            <VirtualizerGridLayout
                items={mockedItems}
                ariaLabel={'test list'}
                mediaState={{ get: vi.fn() } as unknown as MediaState}
                selectionMode={'single'}
                layoutOptions={mockedLayoutOptions}
                isLoadingMore={false}
                onLoadMore={mockedLoadingMore}
                contentItem={(item) => <div>{item.name}</div>}
            />
        );

        expect(mockedLoadingMore).toHaveBeenCalledOnce();
        mockedItems.forEach((item) => expect(screen.getByRole('option', { name: item.name })).toBeVisible());
    });

    it('renders with empty items', () => {
        render(
            <VirtualizerGridLayout
                items={[]}
                ariaLabel={'empty list'}
                mediaState={{ get: vi.fn() } as unknown as MediaState}
                selectionMode={'single'}
                layoutOptions={mockedLayoutOptions}
                isLoadingMore={false}
                onLoadMore={vi.fn()}
                contentItem={(item) => <div>{item.name}</div>}
            />
        );
        expect(screen.queryAllByRole('option')).toHaveLength(0);
    });

    it('renders loading indicator when isLoadingMore is true', () => {
        const mockedLoadingMore = vi.fn();
        render(
            <VirtualizerGridLayout
                items={mockedItems}
                ariaLabel={'loading list'}
                mediaState={{ get: vi.fn() } as unknown as MediaState}
                selectionMode={'single'}
                layoutOptions={mockedLayoutOptions}
                isLoadingMore={true}
                onLoadMore={mockedLoadingMore}
                contentItem={(item) => <div>{item.name}</div>}
            />
        );

        expect(mockedLoadingMore).not.toHaveBeenCalled();
        expect(screen.getByRole('progressbar')).toBeInTheDocument();
    });
});
