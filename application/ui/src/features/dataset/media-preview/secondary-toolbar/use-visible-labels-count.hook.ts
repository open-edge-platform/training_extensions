// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { RefObject, useEffect, useState } from 'react';

const MAX_WIDTH_PERCENT = 0.67;
const BADGE_GAP = 8; // size-100 gap
const OTHER_COMPONENTS_RESERVATION = 104;

interface UseVisibleLabelsCountProps {
    toolbarRef: RefObject<HTMLDivElement | null>;
    labelsContainerRef: RefObject<HTMLDivElement | null>;
    totalLabels: number;
}

const calculateVisibleLabels = (labelsContainer: HTMLElement, availableWidth: number): number => {
    if (availableWidth <= 0) {
        return Infinity;
    }

    const badgeElements = labelsContainer.querySelectorAll('[data-label-badge]');
    let totalWidth = 0;
    let count = 0;

    badgeElements.forEach((badge, index) => {
        const badgeWidth = (badge as HTMLElement).offsetWidth;
        const gap = index > 0 ? BADGE_GAP : 0;

        if (totalWidth + badgeWidth + gap <= availableWidth - OTHER_COMPONENTS_RESERVATION) {
            totalWidth += badgeWidth + gap;
            count++;
        }
    });

    return Math.max(1, count);
};

export const useVisibleLabelsCount = ({ toolbarRef, labelsContainerRef, totalLabels }: UseVisibleLabelsCountProps) => {
    const [collapsedVisibleCount, setCollapsedVisibleCount] = useState(totalLabels);

    useEffect(() => {
        const calculate = () => {
            const toolbar = toolbarRef.current;
            const labelsContainer = labelsContainerRef.current;

            if (!toolbar || !labelsContainer) return;

            const toolbarWidth = toolbar.offsetWidth;
            if (toolbarWidth === 0) {
                setCollapsedVisibleCount(totalLabels);
                return;
            }

            const maxAllowedWidth = toolbarWidth * MAX_WIDTH_PERCENT;

            const children = Array.from(toolbar.children).filter(
                (child) => child.id !== 'labels-container'
            ) as HTMLElement[];
            let otherChildrenWidth = 0;

            children.forEach((child) => {
                otherChildrenWidth += child.offsetWidth;
            });

            const calculatedWidth = toolbarWidth - otherChildrenWidth;
            // Should be capped at 67% according to design
            const availableWidth = Math.max(0, Math.min(calculatedWidth, maxAllowedWidth));

            setCollapsedVisibleCount(calculateVisibleLabels(labelsContainer, availableWidth));
        };

        const resizeObserver = new ResizeObserver(calculate);

        if (toolbarRef.current) {
            resizeObserver.observe(toolbarRef.current);
        }

        return () => {
            resizeObserver.disconnect();
        };
    }, [toolbarRef, labelsContainerRef, totalLabels]);

    return { collapsedVisibleCount };
};
