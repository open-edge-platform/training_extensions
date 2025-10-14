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

    const assignLabel = (labelId: Key | null) => {
        const selectedLabel = projectLabels.find((label) => label.id === labelId) || projectLabels[0];

        annotationsToUpdate.forEach((annotation) => {
            updateAnnotation({ ...annotation, labels: [selectedLabel as Label] });
        });
    };

    const unAssignLabel = (labelId: Key | null) => {
        annotationsToUpdate.forEach((annotation) => {
            updateAnnotation({
                ...annotation,
                labels: annotation.labels?.filter((label) => label.id !== labelId) as Label[],
            });
        });
    };

    return {
        isHidden,
        projectLabels,
        assignLabel,
        unAssignLabel,
        annotationsToUpdate,
    };
};
