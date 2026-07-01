// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { RefObject } from 'react';

import { useEventListener } from 'hooks/event-listener.hook';

import type { Media } from '../../../../constants/shared-types';

export type UseKeyboardNavigationProps = {
    ref: RefObject<HTMLElement | null>;
    items: Media[];
    selectedIndex: number;
    onSelectedMediaItem: (item: Media) => void;
};

export const useKeyboardNavigation = ({
    ref,
    items,
    selectedIndex,
    onSelectedMediaItem,
}: UseKeyboardNavigationProps) => {
    const getNewIndex = (key: 'ArrowUp' | 'ArrowDown' | 'ArrowLeft' | 'ArrowRight') => {
        if (key === 'ArrowUp' || key === 'ArrowLeft') {
            return Math.max(0, selectedIndex - 1);
        }

        if (key === 'ArrowDown' || key === 'ArrowRight') {
            return Math.min(items.length - 1, selectedIndex + 1);
        }

        return selectedIndex;
    };

    useEventListener(
        'keydown',
        (event) => {
            if (
                event.key === 'ArrowUp' ||
                event.key === 'ArrowDown' ||
                event.key === 'ArrowLeft' ||
                event.key === 'ArrowRight'
            ) {
                const newIndex = getNewIndex(event.key);

                items[newIndex] && onSelectedMediaItem(items[newIndex]);
            }
        },
        ref
    );
};
