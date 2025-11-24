// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key } from 'react';

import type { Label } from 'src/constants/shared-types';
import { useProjectLabels } from 'src/features/annotator/hooks/use-project-labels.hook';
import { useAnnotationActions } from 'src/shared/annotator/annotation-actions-provider.component';
import { useSelectedAnnotations } from 'src/shared/annotator/select-annotation-provider.component';

export const useSecondaryToolbarState = () => {
    const { selectedAnnotations } = useSelectedAnnotations();
    const { annotations, updateAnnotations } = useAnnotationActions();

    const projectLabels = useProjectLabels();

    const isHidden = selectedAnnotations.size === 0;

    const annotationsToUpdate = annotations.filter((annotation) => selectedAnnotations.has(annotation.id));

    const addLabels = (labelId: Key | null) => {
        const selectedLabel = projectLabels.find((label) => label.id === labelId);

        if (!selectedLabel) {
            return;
        }

        const updatedAnnotations = annotationsToUpdate.map((annotation) => {
            const hasLabel = annotation.labels?.some((label) => label.id === labelId);

            if (hasLabel) {
                return annotation;
            }

            return {
                ...annotation,
                labels: [...(annotation.labels || []), selectedLabel as Label],
            };
        });

        updateAnnotations(updatedAnnotations);
    };

    const removeLabels = (labelId: Key | null) => {
        const updatedAnnotations = annotationsToUpdate.map((annotation) => ({
            ...annotation,
            labels: annotation.labels?.filter((label) => label.id !== labelId) as Label[],
        }));

        updateAnnotations(updatedAnnotations);
    };

    const toggleLabels = (labelId: Key | null) => {
        const selectedLabel = projectLabels.find((label) => label.id === labelId);

        if (!selectedLabel) {
            return;
        }

        const labelIsAssignedToEveryAnnotation = annotationsToUpdate.every((annotation) =>
            annotation.labels?.some((label) => label.id === labelId)
        );

        if (labelIsAssignedToEveryAnnotation) {
            removeLabels(labelId);
        } else {
            addLabels(labelId);
        }
    };

    return {
        isHidden,

        projectLabels,
        toggleLabels,
        addLabels,
        removeLabels,

        annotationsToUpdate,
    };
};
