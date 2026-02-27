// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useMemo, useRef } from 'react';

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEqual } from 'lodash-es';
import { v4 as uuid } from 'uuid';

import { $api } from '../../api/client';
import type { AnnotationDTO, Label, Media } from '../../constants/shared-types';
import { UndoRedoProvider } from '../../features/dataset/media-preview/primary-toolbar/undo-redo/undo-redo-provider.component';
import useUndoRedoState from '../../features/dataset/media-preview/primary-toolbar/undo-redo/use-undo-redo-state';
import { AnnotatorMode } from '../../features/dataset/media-preview/secondary-toolbar/annotator-modes/mode';
import type { Annotation, Shape } from '../types';
import { EMPTY_LABEL_ID, useProjectLabelsWithEmptyLabel } from './labels';

const mapServerAnnotationsToLocal = (serverAnnotations: AnnotationDTO[], projectLabels: Label[]): Annotation[] => {
    const labelMap = new Map(projectLabels.map((label) => [label.id, label]));

    return serverAnnotations.map((annotation) => {
        // We only get the ids of the labels
        const labels = (annotation.labels ?? [])
            .map((labelRef) => labelMap.get(labelRef.id))
            .filter((label): label is Label => label !== undefined)
            .map((label, idx) => ({ ...label, probability: annotation.confidences?.at(idx) }));

        return {
            ...annotation,
            id: uuid(),
            labels,
        };
    });
};

const mapLocalAnnotationsToServer = (localAnnotations: Annotation[]): AnnotationDTO[] => {
    return localAnnotations.map((annotation) => ({
        // We only want to send the ids of the labels
        labels: annotation.labels.map((label) => ({ id: label.id })),
        shape: annotation.shape,
        ...(annotation.confidences !== undefined && { confidences: annotation.confidences }),
    }));
};

interface AnnotationsContextValue {
    annotations: Annotation[];
    addAnnotations: (shapes: Shape[], labels: Label[]) => string[];
    addAnnotationWithEmptyLabel: (label: Label) => void;
    deleteAnnotations: (annotationIds: string[]) => void;
    updateAnnotations: (updatedAnnotations: Annotation[], labels?: Label[]) => void;
    submitAnnotations: () => Promise<void>;
    isUserReviewed: boolean;
    isSaving: boolean;
    isReadOnlyMode: boolean;
}

const AnnotationsContext = createContext<AnnotationsContextValue | null>(null);

type AnnotationActionsProviderProps = {
    children: ReactNode;
    initialAnnotationsDTO: AnnotationDTO[];
    initialPredictionsDTO: AnnotationDTO[];
    isUserReviewed?: boolean;
    mediaItem: Media;
    mode: AnnotatorMode;
    isReadOnly?: boolean;
};

const filterOutAnnotationWithEmptyLabel = (annotations: Annotation[]): Annotation[] => {
    return annotations.filter((annotation) => annotation.labels.some((label) => label.id !== EMPTY_LABEL_ID));
};

export const AnnotationActionsProvider = ({
    children,
    initialAnnotationsDTO,
    initialPredictionsDTO,
    isUserReviewed = false,
    mediaItem,
    mode,
    isReadOnly = false,
}: AnnotationActionsProviderProps) => {
    const projectId = useProjectIdentifier();
    const saveMutation = $api.useMutation('post', '/api/projects/{project_id}/dataset/media/{media_id}/annotations', {
        meta: {
            invalidateQueries: [
                [
                    'get',
                    '/api/projects/{project_id}/dataset/media/{media_id}/annotations',
                    { params: { path: { project_id: projectId, media_id: mediaItem.id } } },
                ],
                ['get', '/api/projects/{project_id}/dataset/items', { params: { path: { project_id: projectId } } }],
            ],
        },
    });

    const projectLabels = useProjectLabelsWithEmptyLabel();

    const predictions = useMemo(() => {
        return mapServerAnnotationsToLocal(initialPredictionsDTO, projectLabels);
    }, [initialPredictionsDTO, projectLabels]);

    const [annotations, setAnnotations, undoRedoActions] = useUndoRedoState<Annotation[]>(() => {
        return mapServerAnnotationsToLocal(initialAnnotationsDTO, projectLabels);
    });

    const prevInitialAnnotationsDTORef = useRef(initialAnnotationsDTO);

    if (!isEqual(prevInitialAnnotationsDTORef.current, initialAnnotationsDTO)) {
        setAnnotations(mapServerAnnotationsToLocal(initialAnnotationsDTO, projectLabels), true);
        prevInitialAnnotationsDTORef.current = initialAnnotationsDTO;
    }

    const updateAnnotations = (updatedAnnotations: Annotation[], labels?: Label[]) => {
        if (labels !== undefined) {
            const idsToUpdate = new Set(updatedAnnotations.map((a) => a.id));
            setAnnotations((prevAnnotations) =>
                prevAnnotations.map((annotation) =>
                    idsToUpdate.has(annotation.id) ? { ...annotation, labels } : annotation
                )
            );
        } else {
            const updatedMap = new Map(updatedAnnotations.map((annotation) => [annotation.id, annotation]));
            setAnnotations((prevAnnotations) =>
                prevAnnotations.map((annotation) => updatedMap.get(annotation.id) ?? annotation)
            );
        }
    };

    const addAnnotations = (shapes: Shape[], labels: Label[]): string[] => {
        const newAnnotations = shapes.map((shape) => ({
            shape,
            id: uuid(),
            labels,
        }));

        setAnnotations((prevAnnotations) => [...prevAnnotations, ...newAnnotations]);

        return newAnnotations.map((annotation) => annotation.id);
    };

    const deleteAllAnnotations = () => {
        setAnnotations([]);
    };

    const addAnnotationWithEmptyLabel = (emptyLabel: Label) => {
        deleteAllAnnotations();
        addAnnotations([{ type: 'full_image' }], [emptyLabel]);
    };

    const deleteAnnotations = (annotationIds: string[]) => {
        setAnnotations((prevAnnotations) =>
            prevAnnotations.filter((annotation) => !annotationIds.includes(annotation.id))
        );
    };

    const saveAnnotations = async (annotationsDTO: AnnotationDTO[]) => {
        await saveMutation.mutateAsync({
            params: { path: { media_id: mediaItem.id, project_id: projectId } },
            body: { annotations: annotationsDTO },
        });
    };

    const submitPredictions = async () => {
        const serverFormattedAnnotationsWithoutConfidences: AnnotationDTO[] = mapLocalAnnotationsToServer(
            predictions
        ).map(({ confidences, ...restOfAnnotation }) => restOfAnnotation);

        await saveAnnotations(serverFormattedAnnotationsWithoutConfidences);
    };

    const submitAnnotations = async () => {
        if (mode === 'prediction') {
            await submitPredictions();
        } else {
            const filteredAnnotations = filterOutAnnotationWithEmptyLabel(annotations);
            const serverAnnotations = mapLocalAnnotationsToServer(filteredAnnotations);

            await saveAnnotations(serverAnnotations);
        }
    };

    const annotationsToRender = mode === 'annotation' ? annotations : predictions;
    const isReadOnlyMode = isReadOnly || mode === 'prediction';

    return (
        <AnnotationsContext.Provider
            value={{
                isUserReviewed,
                annotations: annotationsToRender,

                // Local
                addAnnotations,
                updateAnnotations,
                deleteAnnotations,
                addAnnotationWithEmptyLabel,

                // Remote
                submitAnnotations,

                isSaving: saveMutation.isPending,
                isReadOnlyMode,
            }}
        >
            <UndoRedoProvider state={undoRedoActions}>{children}</UndoRedoProvider>
        </AnnotationsContext.Provider>
    );
};

export const useAnnotationActions = () => {
    const context = useContext(AnnotationsContext);

    if (context === null) {
        throw new Error('useAnnotationActions must be used within "AnnotationActionsProvider"');
    }

    return context;
};
