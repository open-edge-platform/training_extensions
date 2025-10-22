// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { RefObject } from 'react';

import { useEventListener } from 'hooks/event-listener.hook';
import { DatasetItem } from 'src/constants/shared-types';

export type UseKeyboardNavigationProps = {
    ref: RefObject<HTMLElement | null>;
    items: DatasetItem[];
    selectedIndex: number;
    onSelectedMediaItem: (item: DatasetItem) => void;
};

export const useKeyboardNavigation = ({
    ref,
    items,
    selectedIndex,
    onSelectedMediaItem,
}: UseKeyboardNavigationProps) => {
    const getNewIndex = (key: 'ArrowUp' | 'ArrowDown') => {
        if (key === 'ArrowUp') {
            return Math.max(0, selectedIndex - 1);
        }

        if (key === 'ArrowDown') {
            return Math.min(items.length - 1, selectedIndex + 1);
        }
        return selectedIndex;
    };

    useEventListener(
        'keydown',
        (event) => {
            if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
                const newIndex = getNewIndex(event.key);

                items[newIndex] && onSelectedMediaItem(items[newIndex]);
            }
        },
        ref
    );
};
