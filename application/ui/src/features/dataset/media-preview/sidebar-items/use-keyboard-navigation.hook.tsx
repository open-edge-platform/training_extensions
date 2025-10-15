// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEventListener } from 'hooks/event-listener.hook';
import { DatasetItem } from 'src/features/annotator/types';

type useKeyboardNavigationProps = {
    items: DatasetItem[];
    selectedIndex: number;
    onSelectedMediaItem: (item: DatasetItem) => void;
};

export const useKeyboardNavigation = ({ items, selectedIndex, onSelectedMediaItem }: useKeyboardNavigationProps) => {
    const getNewIndex = (key: 'ArrowUp' | 'ArrowDown') => {
        if (key === 'ArrowUp') {
            return Math.max(0, selectedIndex - 1);
        }

        if (key === 'ArrowDown') {
            return Math.min(items.length - 1, selectedIndex + 1);
        }
        return selectedIndex;
    };

    useEventListener('keydown', (event) => {
        if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
            event.preventDefault();

            const newIndex = getNewIndex(event.key);

            items[newIndex] && onSelectedMediaItem(items[newIndex]);
        }
    });
};
