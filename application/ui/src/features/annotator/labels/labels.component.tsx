// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CSSProperties, useRef, useState } from 'react';

import { ActionButton, Flex, Text } from '@geti/ui';
import { clsx } from 'clsx';

import type { Label } from '../../../constants/shared-types';
import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';
import { useAnnotator } from '../../../shared/annotator/annotator-provider.component';
import { useSelectedAnnotations } from '../../../shared/annotator/select-annotation-provider.component';
import type { Annotation } from '../../../shared/types';
import { useVisibleLabelsCount } from './use-visible-labels-count.hook';

import classes from './labels.module.scss';

interface LabelBadgeProps {
    label: Label;
    isSelected: boolean;
    isHidden: boolean;
    onClick: () => void;
}

const LabelBadge = ({ label, isSelected, isHidden, onClick }: LabelBadgeProps) => {
    return (
        <button
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
    const { collapsedVisibleCount } = useVisibleLabelsCount({ containerRef, totalLabels: labels.length });
    const [isExpanded, setIsExpanded] = useState(false);

    const handleLabelClick = (label: Label) => {
        if (selectedAnnotations.size > 0) {
            const selectedAnnotationsList = annotations.filter((a) => selectedAnnotations.has(a.id));

            const allAnnotationsHaveLabel = selectedAnnotationsList.every((annotation) =>
                annotation.labels.some((l) => l.id === label.id)
            );

            if (allAnnotationsHaveLabel) {
                // Remove label
                const updatedAnnotations = selectedAnnotationsList.map((annotation) => {
                    const filteredLabels = annotation.labels.filter((l) => l.id !== label.id);
                    return { ...annotation, labels: filteredLabels } as Annotation;
                });
                updateAnnotations(updatedAnnotations);
                setSelectedLabelId(null);
            } else {
                // Add label
                updateAnnotations(selectedAnnotationsList, [label]);
                setSelectedLabelId(label.id);
            }
        } else {
            setSelectedLabelId(label.id);
        }
    };

    const handleToggleExpand = () => {
        setIsExpanded((prev) => !prev);
    };

    const hasOverflow = collapsedVisibleCount < labels.length;
    const hiddenCount = labels.length - collapsedVisibleCount;

    return (
        <Flex alignItems='start' gap='size-100' minWidth={0} flex='1'>
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
        </Flex>
    );
};
