// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CSSProperties, Ref, useState } from 'react';

import { ActionButton, Flex, PressableElement, Text, Tooltip, TooltipTrigger } from '@geti/ui';
import { clsx } from 'clsx';
import { isEmpty } from 'lodash-es';

import type { Label } from '../../../constants/shared-types';
import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';
import { useAnnotator } from '../../../shared/annotator/annotator-provider.component';
import { useSelectedAnnotations } from '../../../shared/annotator/select-annotation-provider.component';
import type { Annotation } from '../../../shared/types';
import { toggleLabel } from '../../dataset/media-preview/secondary-toolbar/util';

import classes from './labels.module.scss';

interface LabelBadgeProps {
    label: Label;
    isSelected: boolean;
    isHidden: boolean;
    isDisabled: boolean;
    onClick: () => void;
}

const LabelBadge = ({ label, isSelected, isHidden, isDisabled, onClick }: LabelBadgeProps) => {
    return (
        <TooltipTrigger delay={300}>
            <PressableElement>
                <button
                    onClick={onClick}
                    style={{ '--labelBgColor': label.color } as CSSProperties}
                    className={clsx(classes.badge, {
                        [classes.selected]: isSelected,
                        [classes.hidden]: isHidden,
                        [classes.disabled]: isDisabled,
                    })}
                    aria-pressed={isSelected}
                    aria-label={`Label ${label.name}`}
                    aria-hidden={isHidden}
                    aria-disabled={isDisabled}
                    tabIndex={isHidden ? -1 : 0}
                    data-label-badge
                >
                    <Text UNSAFE_className={classes.badgeText}>{label.name}</Text>
                </button>
            </PressableElement>
            <Tooltip>{label.name}</Tooltip>
        </TooltipTrigger>
    );
};

interface LabelsProps {
    ref?: Ref<HTMLDivElement>;
    collapsedVisibleCount?: number;
    isClassification?: boolean;
    isMultiLabel?: boolean;
    isReadOnly?: boolean;
}

export const Labels = ({
    ref,
    collapsedVisibleCount = Infinity,
    isClassification = false,
    isMultiLabel = false,
    isReadOnly = false,
}: LabelsProps) => {
    const { selectedLabelId, setSelectedLabelId, labels } = useAnnotator();
    const { selectedAnnotations } = useSelectedAnnotations();
    const { annotations, addAnnotations, updateAnnotations, deleteAnnotations } = useAnnotationActions();

    const [isExpanded, setIsExpanded] = useState(false);

    const handleClassificationClick = (label: Label) => {
        if (isReadOnly) return;

        if (isEmpty(annotations)) {
            addAnnotations([{ type: 'full_image' }], [label]);
            return;
        }

        if (isMultiLabel) {
            const updatedAnnotations = annotations.map((annotation) => ({
                ...annotation,
                labels: toggleLabel(label, annotation.labels),
            }));

            const hasNoLabels = updatedAnnotations.every(({ labels: annotationLabels }) => isEmpty(annotationLabels));

            if (hasNoLabels) {
                deleteAnnotations(updatedAnnotations.map(({ id }) => id));
            } else {
                updateAnnotations(updatedAnnotations);
            }
        } else {
            const isAlreadySelected = annotations.some((annotation) =>
                annotation.labels.some((l) => l.id === label.id)
            );

            if (isAlreadySelected) {
                deleteAnnotations(annotations.map(({ id }) => id));
            } else {
                updateAnnotations(annotations.map((annotation) => ({ ...annotation, labels: [label] })));
            }
        }
    };

    const handleNonClassificationClick = (label: Label) => {
        if (isReadOnly) return;
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

    const handleLabelClick = (label: Label) => {
        if (isClassification) {
            handleClassificationClick(label);
        } else {
            handleNonClassificationClick(label);
        }
    };

    const handleToggleExpand = () => {
        setIsExpanded((prev) => !prev);
    };

    const isLabelSelected = (label: Label): boolean => {
        if (isClassification) {
            return annotations.some((annotation) => annotation.labels.some((l) => l.id === label.id));
        }
        return selectedLabelId === label.id;
    };

    const hasOverflow = collapsedVisibleCount < labels.length;
    const hiddenCount = labels.length - collapsedVisibleCount;

    return (
        <Flex alignItems='start' gap='size-100' minWidth={0} flex='1'>
            <div
                ref={ref}
                aria-label={'Labels'}
                className={clsx(classes.labelsContainer, {
                    [classes.expanded]: isExpanded,
                    [classes.readOnlyLabels]: isReadOnly,
                })}
                aria-disabled={isReadOnly}
            >
                {labels.map((label, index) => (
                    <LabelBadge
                        key={label.id}
                        label={label}
                        isSelected={isLabelSelected(label)}
                        isHidden={!isExpanded && index >= collapsedVisibleCount}
                        isDisabled={isReadOnly}
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
