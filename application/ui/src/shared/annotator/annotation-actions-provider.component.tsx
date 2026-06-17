// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useMemo, useRef } from 'react';

import { useQueryClient } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEqual } from 'lodash-es';
import { v4 as uuid } from 'uuid';

import { $api } from '../../api/client';
import type { AnnotationDTO, DatasetSubset, Label, Media } from '../../constants/shared-types';
import { UndoRedoProvider } from '../../features/dataset/media-preview/primary-toolbar/undo-redo/undo-redo-provider.component';
import useUndoRedoState from '../../features/dataset/media-preview/primary-toolbar/undo-redo/use-undo-redo-state';
import { isVideoFrame } from '../media-item-utils';
import type { Annotation, AnnotationLabelRef, Shape } from '../types';
import { isNonEmptyArray } from '../util';
import { mapLocalAnnotationsToServer, mapServerAnnotationsToLocal } from './annotation-mappers';
import type { AnnotatorMode } from './annotator-mode';
import { EMPTY_LABEL_ID, isNonEmptyLabel, useProjectLabelsWithEmptyLabel } from './labels';
import { incrementCachedAnnotatedFrameCount } from './util';

type AnnotationsContextValue = {
    annotations: Annotation[];
    canSubmit: boolean;
    hasInvalidAnnotation: boolean;
    addAnnotations: (shapes: Shape[], labels: AnnotationLabelRef[]) => string[];
    addAnnotationWithEmptyLabel: (label: Label) => void;
    deleteAnnotations: (annotationIds: string[]) => void;
    updateAnnotations: (updatedAnnotations: Annotation[], labels?: AnnotationLabelRef[]) => void;
    submitAnnotations: (subset: DatasetSubset) => Promise<void>;
    submitPredictions: (subset: DatasetSubset) => Promise<void>;
    resetAnnotations: () => void;
    replaceAnnotations: (annotations: Annotation[]) => void;
    isUserReviewed: boolean;
    isSaving: boolean;
    isReadOnlyMode: boolean;
    initialAnnotations: Annotation[];
    initialPredictions: Annotation[];
};

const AnnotationsContext = createContext<AnnotationsContextValue | null>(null);

export type AnnotationActionsProviderProps = {
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
    const queryClient = useQueryClient();
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
            ],
        },
    });

    const projectLabels = useProjectLabelsWithEmptyLabel();

    const predictions = useMemo(() => {
        return mapServerAnnotationsToLocal(initialPredictionsDTO);
    }, [initialPredictionsDTO]);

    const initialAnnotations = useMemo(() => {
        return mapServerAnnotationsToLocal(initialAnnotationsDTO);
    }, [initialAnnotationsDTO]);

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

    const updateAnnotations = (updatedAnnotations: Annotation[], labels?: AnnotationLabelRef[]) => {
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

    const addAnnotations = (shapes: Shape[], labels: AnnotationLabelRef[]): string[] => {
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
        addAnnotations([{ type: 'full_image' }], [{ id: emptyLabel.id }]);
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
        const query = isVideoFrame(mediaItem) ? { frame_index: mediaItem.frame_number } : undefined;

        await saveMutation
            .mutateAsync({
                params: { path: { media_id: mediaItem.id, project_id: projectId }, query },
                body: { annotations: annotationsDTO, subset: subset ?? undefined },
            })
            .then(() => {
                if (isVideoFrame(mediaItem)) {
                    incrementCachedAnnotatedFrameCount(queryClient, mediaItem);
                }
            });

        undoRedoActions.reset(mapServerAnnotationsToLocal(annotationsDTO));
    };

    const submitPredictions = async (subset: DatasetSubset) => {
        const validLabelIds = new Set(projectLabels.map((l) => l.id));
        const serverFormattedAnnotationsWithoutConfidences = mapLocalAnnotationsToServer(predictions, validLabelIds)
            .map(({ confidences, ...restOfAnnotation }) => restOfAnnotation)
            .filter((annotation) => isNonEmptyArray(annotation.labels) && annotation.labels.every(isNonEmptyLabel));

        await saveAnnotations(serverFormattedAnnotationsWithoutConfidences, subset);
    };

    const annotationsToRender = mode === 'annotation' ? annotations : predictions;

    const submitAnnotations = async (subset: DatasetSubset) => {
        const validLabelIds = new Set(projectLabels.map((l) => l.id));
        const filteredAnnotations = filterOutAnnotationWithEmptyLabel(annotations);
        const serverAnnotations = mapLocalAnnotationsToServer(filteredAnnotations, validLabelIds);

        await saveAnnotations(serverAnnotations, subset);
    };

    const hasChangedAnnotations = useMemo(() => {
        const filteredAnnotations = filterOutAnnotationWithEmptyLabel(annotations);
        const currentServerAnnotations = mapLocalAnnotationsToServer(filteredAnnotations);

        return !isEqual(currentServerAnnotations, initialAnnotationsDTO);
    }, [annotations, initialAnnotationsDTO]);

    const hasEmptyLabelSelection = useMemo(() => {
        return annotations.some((annotation) => annotation.labels.some((label) => label.id === EMPTY_LABEL_ID));
    }, [annotations]);

    const hasInvalidAnnotation = useMemo(() => {
        return annotations.some((annotation) => annotation.labels.length === 0);
    }, [annotations]);

    const canSubmit =
        mode === 'prediction'
            ? predictions.length > 0
            : !hasInvalidAnnotation && (hasChangedAnnotations || hasEmptyLabelSelection);
    const isReadOnlyMode = isReadOnly || mode === 'prediction';

    return (
        <AnnotationsContext.Provider
            value={{
                isUserReviewed,
                annotations: annotationsToRender,
                canSubmit,
                hasInvalidAnnotation,

                // Local
                addAnnotations,
                updateAnnotations,
                deleteAnnotations,
                addAnnotationWithEmptyLabel,
                resetAnnotations,
                replaceAnnotations,
                initialAnnotations,
                initialPredictions: predictions,

                // Remote
                submitAnnotations,
                submitPredictions,

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
