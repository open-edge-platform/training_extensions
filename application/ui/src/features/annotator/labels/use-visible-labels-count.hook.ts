// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { RefObject, useEffect, useState } from 'react';

import classes from './labels.module.scss';

const BADGE_GAP = 8; // size-100 gap

const calculateVisibleLabels = (container: HTMLDivElement): number => {
    const containerWidth = container.offsetWidth;
    const badgeElements = container.querySelectorAll('[data-label-badge]');
    const hiddenElement = container.querySelectorAll(`.${classes.hidden}`).length > 0;
    // "Show more" button width is 80px when visible, 20px reserved space for smooth transition
    const showMoreReservedWidth = hiddenElement ? 20 : 80;
    let totalWidth = 0;
    let count = 0;

    badgeElements.forEach((badge, index) => {
        const badgeWidth = (badge as HTMLElement).offsetWidth;
        const gap = index > 0 ? BADGE_GAP : 0;

        if (totalWidth + badgeWidth + gap <= containerWidth - showMoreReservedWidth) {
            totalWidth += badgeWidth + gap;
            count++;
        }
    });

    return Math.max(1, count);
};

interface UseVisibleLabelsCountProps {
    containerRef: RefObject<HTMLDivElement | null>;
    totalLabels: number;
}

export const useVisibleLabelsCount = ({ containerRef, totalLabels }: UseVisibleLabelsCountProps) => {
    const [collapsedVisibleCount, setCollapsedVisibleCount] = useState(totalLabels);

    useEffect(() => {
        const updateVisibleCount = () => {
            const container = containerRef.current;
            if (!container) return;

            setCollapsedVisibleCount(calculateVisibleLabels(container));
        };

        const resizeObserver = new ResizeObserver(updateVisibleCount);
        if (containerRef.current) {
            resizeObserver.observe(containerRef.current);
        }

        return () => {
            resizeObserver.disconnect();
        };
    }, [containerRef, totalLabels]);

    return { collapsedVisibleCount };
};
