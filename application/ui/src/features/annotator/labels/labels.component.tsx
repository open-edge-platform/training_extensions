// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CSSProperties, useEffect, useRef, useState } from 'react';

import { ActionButton, Text } from '@geti/ui';
import { clsx } from 'clsx';

import type { Label } from '../../../constants/shared-types';
import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';
import { useAnnotator } from '../../../shared/annotator/annotator-provider.component';
import { useSelectedAnnotations } from '../../../shared/annotator/select-annotation-provider.component';

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

interface LabelBadgeProps {
    label: Label;
    isSelected: boolean;
    isHidden: boolean;
    onClick: () => void;
}

const LabelBadge = ({ label, isSelected, isHidden, onClick }: LabelBadgeProps) => {
    return (
        <button
            type={'button'}
            onClick={onClick}
            style={{ '--labelBgColor': label.color } as CSSProperties}
            className={clsx(classes.badge, { [classes.selected]: isSelected, [classes.hidden]: isHidden })}
            aria-pressed={isSelected}
            aria-label={`Label ${label.name}`}
            aria-hidden={isHidden}
            tabIndex={isHidden ? -1 : 0}
            data-label-badge
        >
            <Text UNSAFE_className={classes.badgeText}>{label.name}</Text>
        </button>
    );
};

export const Labels = () => {
    const { selectedLabelId, setSelectedLabelId, labels } = useAnnotator();
    const { selectedAnnotations } = useSelectedAnnotations();
    const { annotations, updateAnnotations } = useAnnotationActions();

    const containerRef = useRef<HTMLDivElement>(null);
    const [collapsedVisibleCount, setCollapsedVisibleCount] = useState(labels.length);
    const [isExpanded, setIsExpanded] = useState(false);

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
    }, [labels]);

    const handleLabelClick = (label: Label) => {
        setSelectedLabelId(label.id);

        if (selectedAnnotations.size > 0) {
            const updatedAnnotations = annotations
                .filter((annotation) => selectedAnnotations.has(annotation.id))
                .map((annotation) => ({
                    ...annotation,
                    labels: [label],
                }));

            if (updatedAnnotations.length > 0) {
                updateAnnotations(updatedAnnotations);
            }
        }
    };

    const handleToggleExpand = () => {
        setIsExpanded((prev) => !prev);
    };

    const hasOverflow = collapsedVisibleCount < labels.length;
    const hiddenCount = labels.length - collapsedVisibleCount;

    return (
        <div className={classes.wrapper}>
            <div ref={containerRef} className={clsx(classes.labelsContainer, { [classes.expanded]: isExpanded })}>
                {labels.map((label, index) => (
                    <LabelBadge
                        key={label.id}
                        label={label}
                        isSelected={selectedLabelId === label.id}
                        isHidden={!isExpanded && index >= collapsedVisibleCount}
                        onClick={() => handleLabelClick(label)}
                    />
                ))}
            </div>
            {hasOverflow && (
                <ActionButton
                    isQuiet
                    onPress={handleToggleExpand}
                    aria-expanded={isExpanded}
                    UNSAFE_className={classes.showMoreButton}
                >
                    {isExpanded ? 'Show less' : `+${hiddenCount} more`}
                </ActionButton>
            )}
        </div>
    );
};
