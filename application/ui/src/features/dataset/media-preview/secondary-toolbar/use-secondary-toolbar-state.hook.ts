// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key } from 'react';

import { useAnnotationActions } from 'src/features/annotator/annotation-actions-provider.component';
import { useAnnotator } from 'src/features/annotator/annotator-provider.component';
import { useProjectLabels } from 'src/features/annotator/hooks/use-project-labels.hook';
import { useSelectedAnnotations } from 'src/features/annotator/select-annotation-provider.component';
import { Label } from 'src/features/annotator/types';

export const useSecondaryToolbarState = () => {
    const { activeTool } = useAnnotator();
    const { selectedAnnotations } = useSelectedAnnotations();
    const { annotations, updateAnnotation } = useAnnotationActions();

    const projectLabels = useProjectLabels();

    const isHidden = selectedAnnotations.size === 0 && activeTool === 'selection';

    const annotationsToUpdate = annotations.filter((annotation) => selectedAnnotations.has(annotation.id));

    const addLabels = (labelId: Key | null) => {
        const selectedLabel = projectLabels.find((label) => label.id === labelId);

        annotationsToUpdate.forEach((annotation) => {
            const hasLabel = annotation.labels?.some((label) => label.id === labelId);

            if (!hasLabel) {
                updateAnnotation({
                    ...annotation,
                    labels: [...(annotation.labels || []), selectedLabel as Label],
                });
            }
        });
    };

    const removeLabels = (labelId: Key | null) => {
        annotationsToUpdate.forEach((annotation) => {
            updateAnnotation({
                ...annotation,
                labels: annotation.labels?.filter((label) => label.id !== labelId) as Label[],
            });
        });
    };

    const toggleLabel = (labelId: Key | null) => {
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
        toggleLabel,
        annotationsToUpdate,
    };
};
