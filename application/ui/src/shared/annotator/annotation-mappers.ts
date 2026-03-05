// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { v4 as uuid } from 'uuid';

import type { AnnotationDTO, Label } from '../../constants/shared-types';
import { Annotation, AnnotationLabel } from '../types';

export const mapServerAnnotationsToLocal = (
    serverAnnotations: AnnotationDTO[],
    projectLabels: Label[]
): Annotation[] => {
    const labelMap = new Map(projectLabels.map((label) => [label.id, label]));

    return serverAnnotations.map((annotation) => {
        // We only get the ids of the labels
        const labels = (annotation.labels ?? [])
            .map((labelRef, idx) => {
                const label = labelMap.get(labelRef.id);

                if (label === undefined) {
                    return undefined;
                }

                const probability = annotation.confidences?.at(idx);

                if (probability === undefined) {
                    return label;
                }

                return {
                    ...label,
                    probability,
                };
            })
            .filter((label): label is AnnotationLabel => label !== undefined);

        return {
            ...annotation,
            id: uuid(),
            labels,
        };
    });
};

export const mapLocalAnnotationsToServer = (localAnnotations: Annotation[]): AnnotationDTO[] => {
    return localAnnotations.map((annotation) => ({
        // We only want to send the ids of the labels
        labels: annotation.labels.map((label) => ({ id: label.id })),
        shape: annotation.shape,
        ...(annotation.confidences !== undefined && { confidences: annotation.confidences }),
    }));
};
