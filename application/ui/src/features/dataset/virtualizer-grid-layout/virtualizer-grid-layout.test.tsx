// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@testing-library/react';
import { Size } from 'react-aria-components';
import { components } from 'src/api/openapi-spec';
import { MediaState } from 'src/routes/dataset/provider';
import { describe, expect, it } from 'vitest';

import { VirtualizerGridLayout } from './virtualizer-grid-layout.component';

const mockedItem = (item: Partial<components['schemas']['DatasetItem']>): components['schemas']['DatasetItem'] => ({
    id: '123',
    name: 'test-1',
    format: 'jpg',
    width: 0,
    height: 0,
    size: 0,
    subset: 'unassigned',
    ...item,
});

// required configuration; otherwise, the list renders empty
const mockedLayoutOptions = {
    maxColumns: 1,
    maxItemSize: new Size(100, 100),
    maxHorizontalSpace: 1000,
};

describe('VirtualizerGridLayout', () => {
    const mockedItems = [mockedItem({ id: '111', name: 'test-1' }), mockedItem({ id: '222', name: 'test-2' })];

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
