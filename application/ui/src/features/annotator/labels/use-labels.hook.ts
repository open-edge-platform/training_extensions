// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback } from 'react';

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEmpty } from 'lodash-es';

import { validateLabelName } from '../../../components/label-fields/label-validation';
import type { Label } from '../../../constants/shared-types';
import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';
import { EMPTY_LABEL_ID, filterOutEmptyLabels } from '../../../shared/annotator/labels';
import { useSelectedAnnotations } from '../../../shared/annotator/select-annotation-provider.component';
import type { Annotation } from '../../../shared/types';
import { toggleLabel } from '../../dataset/media-preview/secondary-toolbar/util';
import { useAnnotatorLabels } from '../annotator-labels-provider.component';
import { useUpdateLabel } from './api/use-update-label.hook';

type UseLabelsOptions = {
    isClassification?: boolean;
    isMultiLabel?: boolean;
};

export const useLabels = ({ isClassification = false, isMultiLabel = false }: UseLabelsOptions = {}) => {
    const { selectedLabelId, setSelectedLabelId, labels } = useAnnotatorLabels();
    const { selectedAnnotations } = useSelectedAnnotations();
    const { annotations, addAnnotations, updateAnnotations, addAnnotationWithEmptyLabel } = useAnnotationActions();

    const projectId = useProjectIdentifier();
    const updateLabelMutation = useUpdateLabel();

    const editableLabels = labels.filter((label) => label.id !== EMPTY_LABEL_ID);
    const hasLabels = editableLabels.length > 0;

    const toggleClassificationLabel = (label: Label) => {
        if (isEmpty(annotations)) {
            addAnnotations([{ type: 'full_image' }], [label]);
            return;
        }

        if (label.id === EMPTY_LABEL_ID && annotations.length !== 0) {
            addAnnotationWithEmptyLabel(label);
            return;
        }

        if (isMultiLabel) {
            const hasEmptyLabel = annotations.some((annotation) =>
                annotation.labels.some((l) => l.id === EMPTY_LABEL_ID)
            );

            let annotationsToUpdate = annotations;

            // If there is an annotation with the empty label, and we are trying to add assign new label to it, we
            // need to remove the empty label from the annotation first.
            if (hasEmptyLabel) {
                annotationsToUpdate = annotations.map((annotation) => ({
                    ...annotation,
                    labels: filterOutEmptyLabels(annotation.labels),
                }));
            }

            const updatedAnnotations = annotationsToUpdate.map((annotation) => ({
                ...annotation,
                labels: toggleLabel(label, annotation.labels),
            }));

            updateAnnotations(updatedAnnotations);
        } else {
            const isAlreadySelected = annotations.some((annotation) =>
                annotation.labels.some((l) => l.id === label.id)
            );

            if (isAlreadySelected) {
                updateAnnotations(annotations.map((annotation) => ({ ...annotation, labels: [] })));
            } else {
                updateAnnotations(annotations.map((annotation) => ({ ...annotation, labels: [label] })));
            }
        }
    };

    const toggleDetectionLabel = (label: Label) => {
        if (label.id === EMPTY_LABEL_ID) {
            addAnnotationWithEmptyLabel(label);
            return;
        }

        const selectedAnnotationsList = annotations.filter((a) => selectedAnnotations.has(a.id));

        if (selectedAnnotationsList.length > 0) {
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

    const toggleLabelOnAnnotations = (label: Label) => {
        if (isClassification) {
            toggleClassificationLabel(label);
        } else {
            toggleDetectionLabel(label);
        }
    };

    const isLabelActive = (label: Label): boolean => {
        if (label.id === EMPTY_LABEL_ID) {
            return false;
        }

        if (isClassification) {
            return annotations.some((annotation) => annotation.labels.some((l) => l.id === label.id));
        }

        return selectedLabelId === label.id;
    };

    const addLabel = (name: string, color: string) => {
        updateLabelMutation.mutate({
            body: {
                labels_to_add: [{ name, color, hotkey: null }],
            },
            params: {
                path: {
                    project_id: projectId,
                },
            },
        });
    };

    const updateLabel = useCallback(
        (labelId: string, updates: { name: string; color: string; hotkey: string | null | undefined }) => {
            updateLabelMutation.mutate({
                body: {
                    labels_to_edit: [
                        { id: labelId, new_name: updates.name, new_color: updates.color, new_hotkey: updates.hotkey },
                    ],
                },
                params: {
                    path: {
                        project_id: projectId,
                    },
                },
            });
        },
        [updateLabelMutation, projectId]
    );

    const deleteLabel = (labelId: string) => {
        updateLabelMutation.mutate({
            body: {
                labels_to_remove: [{ id: labelId }],
            },
            params: {
                path: {
                    project_id: projectId,
                },
            },
        });
    };

    const validateName = (name: string, excludeId?: string): string | undefined => {
        return validateLabelName(name, editableLabels, excludeId);
    };

    return {
        labels,
        editableLabels,
        hasLabels,
        selectedLabelId,
        isUpdating: updateLabelMutation.isPending,
        addLabel,
        updateLabel,
        deleteLabel,
        toggleLabelOnAnnotations,
        isLabelActive,
        validateName,
    };
};
