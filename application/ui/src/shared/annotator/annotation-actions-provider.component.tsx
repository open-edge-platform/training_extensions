// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useMemo, useRef } from 'react';

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEqual } from 'lodash-es';
import { v4 as uuid } from 'uuid';

import { $api } from '../../api/client';
import type { AnnotationDTO, DatasetSubset, Label, Media } from '../../constants/shared-types';
import { UndoRedoProvider } from '../../features/dataset/media-preview/primary-toolbar/undo-redo/undo-redo-provider.component';
import useUndoRedoState from '../../features/dataset/media-preview/primary-toolbar/undo-redo/use-undo-redo-state';
import { isVideoFrame } from '../media-item-utils';
import type { Annotation, Shape } from '../types';
import { mapLocalAnnotationsToServer, mapServerAnnotationsToLocal } from './annotation-mappers';
import type { AnnotatorMode } from './annotator-mode';
import { EMPTY_LABEL_ID, useProjectLabelsWithEmptyLabel } from './labels';

type AnnotationsContextValue = {
    annotations: Annotation[];
    canSubmit: boolean;
    addAnnotations: (shapes: Shape[], labels: Label[]) => string[];
    addAnnotationWithEmptyLabel: (label: Label) => void;
    deleteAnnotations: (annotationIds: string[]) => void;
    updateAnnotations: (updatedAnnotations: Annotation[], labels?: Label[]) => void;
    submitAnnotations: (subset: DatasetSubset) => Promise<void>;
    resetAnnotations: () => void;
    replaceAnnotations: (annotations: Annotation[]) => void;
    isUserReviewed: boolean;
    isSaving: boolean;
    isReadOnlyMode: boolean;
};

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
                [
                    'get',
                    '/api/projects/{project_id}/dataset/items/{dataset_item_id}',
                    { params: { path: { project_id: projectId, dataset_item_id: mediaItem.id } } },
                ],
                [
                    'get',
                    '/api/projects/{project_id}/dataset/media/{media_id}/frames',
                    { params: { path: { project_id: projectId, media_id: mediaItem.id } } },
                ],
                ['get', '/api/projects/{project_id}/dataset/media', { params: { path: { project_id: projectId } } }],
            ],
        },
    });

    const projectLabels = useProjectLabelsWithEmptyLabel();

    const predictions = useMemo(() => {
        return mapServerAnnotationsToLocal(initialPredictionsDTO, projectLabels);
    }, [initialPredictionsDTO, projectLabels]);

    const initialAnnotations = useMemo(() => {
        return mapServerAnnotationsToLocal(initialAnnotationsDTO, projectLabels);
    }, [initialAnnotationsDTO, projectLabels]);

    const [annotations, setAnnotations, undoRedoActions] = useUndoRedoState<Annotation[]>(initialAnnotations);

    const resetAnnotations = () => {
        undoRedoActions.reset(initialAnnotations);
    };

    const prevInitialAnnotationsDTORef = useRef(initialAnnotationsDTO);

    // Reset annotations when source annotations change.
    if (!isEqual(prevInitialAnnotationsDTORef.current, initialAnnotationsDTO)) {
        undoRedoActions.reset(initialAnnotations);
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

    const replaceAnnotations = (newAnnotations: Annotation[]) => {
        setAnnotations(() => newAnnotations);
    };

    const saveAnnotations = async (annotationsDTO: AnnotationDTO[], subset?: DatasetSubset) => {
        const query = isVideoFrame(mediaItem)
            ? {
                  frame_index: mediaItem.frame_number,
              }
            : undefined;

        await saveMutation.mutateAsync({
            params: { path: { media_id: mediaItem.id, project_id: projectId }, query },
            body: { annotations: annotationsDTO, subset: subset ?? undefined },
        });

        undoRedoActions.reset(mapServerAnnotationsToLocal(annotationsDTO, projectLabels));
    };

    const submitPredictions = async (subset: DatasetSubset) => {
        const serverFormattedAnnotationsWithoutConfidences: AnnotationDTO[] = mapLocalAnnotationsToServer(
            predictions
        ).map(({ confidences, ...restOfAnnotation }) => restOfAnnotation);

        await saveAnnotations(serverFormattedAnnotationsWithoutConfidences, subset);
    };

    const submitAnnotations = async (subset: DatasetSubset) => {
        if (mode === 'prediction') {
            await submitPredictions(subset);
        } else {
            const filteredAnnotations = filterOutAnnotationWithEmptyLabel(annotations);
            const serverAnnotations = mapLocalAnnotationsToServer(filteredAnnotations);

            await saveAnnotations(serverAnnotations, subset);
        }
    };

    const hasChangedAnnotations = useMemo(() => {
        const filteredAnnotations = filterOutAnnotationWithEmptyLabel(annotations);
        const currentServerAnnotations = mapLocalAnnotationsToServer(filteredAnnotations);

        return !isEqual(currentServerAnnotations, initialAnnotationsDTO);
    }, [annotations, initialAnnotationsDTO]);

    const hasEmptyLabelSelection = useMemo(() => {
        return annotations.some((annotation) => annotation.labels.some((label) => label.id === EMPTY_LABEL_ID));
    }, [annotations]);

    const canSubmit = mode === 'prediction' ? predictions.length > 0 : hasChangedAnnotations || hasEmptyLabelSelection;

    const annotationsToRender = mode === 'annotation' ? annotations : predictions;
    const isReadOnlyMode = isReadOnly || mode === 'prediction';

    return (
        <AnnotationsContext.Provider
            value={{
                isUserReviewed,
                annotations: annotationsToRender,
                canSubmit,

                // Local
                addAnnotations,
                updateAnnotations,
                deleteAnnotations,
                addAnnotationWithEmptyLabel,
                resetAnnotations,
                replaceAnnotations,

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
