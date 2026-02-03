// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CSSProperties } from 'react';

import { Divider, Flex, Text } from '@geti/ui';
import { clsx } from 'clsx';
import { isEmpty } from 'lodash-es';

import type { Label } from '../../../constants/shared-types';
import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';
import { useAnnotator } from '../../../shared/annotator/annotator-provider.component';
import { EMPTY_LABEL_ID } from '../../../shared/annotator/labels';
import { useSelectedAnnotations } from '../../../shared/annotator/select-annotation-provider.component';
import type { Annotation } from '../../../shared/types';
import { toggleLabel } from '../../dataset/media-preview/secondary-toolbar/util';

import classes from './labels.module.scss';

interface LabelBadgeProps {
    label: Label;
    isSelected: boolean;
    isDisabled: boolean;
    onClick: () => void;
}

const LabelBadge = ({ label, isSelected, isDisabled, onClick }: LabelBadgeProps) => {
    return (
        <button
            onClick={onClick}
            style={{ '--labelBgColor': label.color } as CSSProperties}
            className={clsx(classes.badge, { [classes.selected]: isSelected, [classes.disabled]: isDisabled })}
            aria-pressed={isSelected}
            aria-label={`Label ${label.name}`}
            aria-disabled={isDisabled}
        >
            <Text UNSAFE_className={classes.badgeText}>{label.name}</Text>
        </button>
    );
};

interface LabelsProps {
    isClassification?: boolean;
    isMultiLabel?: boolean;
    isReadOnly?: boolean;
}

const filterOutEmptyLabels = (labels: Label[]): Label[] => labels.filter((label) => label.id !== EMPTY_LABEL_ID);

export const Labels = ({ isClassification = false, isMultiLabel = false, isReadOnly = false }: LabelsProps) => {
    const { selectedLabelId, setSelectedLabelId, labels } = useAnnotator();
    const { selectedAnnotations } = useSelectedAnnotations();
    const { annotations, addAnnotations, updateAnnotations, deleteAnnotations } = useAnnotationActions();

    const handleClassificationClick = (label: Label) => {
        if (isReadOnly) return;

        if (isEmpty(annotations)) {
            addAnnotations([{ type: 'full_image' }], [label]);
            return;
        }

        if (label.id === EMPTY_LABEL_ID && annotations.length !== 0) {
            deleteAnnotations(annotations.map(({ id }) => id));
            addAnnotations([{ type: 'full_image' }], [label]);
            return;
        }

        if (isMultiLabel) {
            const annotationWithoutEmptyLabel = annotations.map((annotation) => ({
                ...annotation,
                labels: filterOutEmptyLabels(annotation.labels),
            }));

            const updatedAnnotations = annotationWithoutEmptyLabel.map((annotation) => ({
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

        if (label.id === EMPTY_LABEL_ID) {
            deleteAnnotations(annotations.map(({ id }) => id));
            addAnnotations([{ type: 'full_image' }], [label]);
            return;
        }

        if (selectedAnnotations.size > 0) {
            const selectedAnnotationsList = annotations.filter((a) => selectedAnnotations.has(a.id));

            const selectedAnnotationsWithoutEmptyLabel: Annotation[] = selectedAnnotationsList.map((annotation) => ({
                ...annotation,
                labels: filterOutEmptyLabels(annotation.labels),
            }));

            const allAnnotationsHaveLabel = selectedAnnotationsWithoutEmptyLabel.every((annotation) =>
                annotation.labels.some((l) => l.id === label.id)
            );

            if (allAnnotationsHaveLabel) {
                // Remove label
                const updatedAnnotations = selectedAnnotationsWithoutEmptyLabel.map((annotation) => {
                    const filteredLabels = annotation.labels.filter((l) => l.id !== label.id);
                    return { ...annotation, labels: filteredLabels } as Annotation;
                });
                updateAnnotations(updatedAnnotations);
                setSelectedLabelId(null);
            } else {
                // Add label
                updateAnnotations(selectedAnnotationsWithoutEmptyLabel, [label]);
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

    const isLabelSelected = (label: Label): boolean => {
        if (isClassification) {
            return annotations.some((annotation) => annotation.labels.some((l) => l.id === label.id));
        }
        return selectedLabelId === label.id;
    };

    return (
        <Flex alignItems='start' gap='size-100' minWidth={0} flex='1'>
            <div
                aria-label={'Labels'}
                className={clsx(classes.labelsContainer, { [classes.readOnlyLabels]: isReadOnly })}
                aria-disabled={isReadOnly}
            >
                {labels.map((label) => (
                    <>
                        {label.id === EMPTY_LABEL_ID && <Divider size={'S'} orientation={'vertical'} />}
                        <LabelBadge
                            key={label.id}
                            label={label}
                            isSelected={isLabelSelected(label)}
                            isDisabled={isReadOnly}
                            onClick={() => handleLabelClick(label)}
                        />
                    </>
                ))}
            </div>
        </Flex>
    );
};
