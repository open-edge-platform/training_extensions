// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useEffect } from 'react';

import { useProject } from 'hooks/api/project.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { v4 as uuid } from 'uuid';

import { $api } from '../../api/client';
import type { components } from '../../api/openapi-spec';
import type { DatasetItem, Label } from '../../constants/shared-types';
import { UndoRedoProvider } from '../../features/dataset/media-preview/primary-toolbar/undo-redo/undo-redo-provider.component';
import useUndoRedoState from '../../features/dataset/media-preview/primary-toolbar/undo-redo/use-undo-redo-state';
import type { Annotation, Shape } from '../types';

type ServerAnnotation = components['schemas']['DatasetItemAnnotation-Input'];

const mapServerAnnotationsToLocal = (serverAnnotations: ServerAnnotation[], projectLabels: Label[]): Annotation[] => {
    const labelMap = new Map(projectLabels.map((label) => [label.id, label]));

    return serverAnnotations.map((annotation) => {
        // We only get the ids of the labels
        const labels = (annotation.labels ?? [])
            .map((labelRef) => labelMap.get(labelRef.id))
            .filter((label): label is Label => label !== undefined);

        return {
            ...annotation,
            id: uuid(),
            labels,
        } as Annotation;
    });
};

const mapLocalAnnotationsToServer = (localAnnotations: Annotation[]): ServerAnnotation[] => {
    return localAnnotations.map((annotation) => ({
        // We only want to send the ids of the labels
        labels: annotation.labels.map((label) => ({ id: label.id })),
        shape: annotation.shape,
        ...(annotation.confidences !== undefined && { confidences: annotation.confidences }),
    }));
};

interface AnnotationsContextValue {
    annotations: Annotation[];
    addAnnotations: (shapes: Shape[], labels: Label[]) => void;
    deleteAnnotations: (annotationIds: string[]) => void;
    updateAnnotations: (updatedAnnotations: Annotation[]) => void;
    submitAnnotations: () => Promise<void>;
    isUserReviewed: boolean;
    isSaving: boolean;
}

const AnnotationsContext = createContext<AnnotationsContextValue | null>(null);

type AnnotationActionsProviderProps = {
    children: ReactNode;
    initialAnnotationsDTO?: ServerAnnotation[];
    isUserReviewed?: boolean;
    mediaItem: DatasetItem;
};

export const AnnotationActionsProvider = ({
    children,
    initialAnnotationsDTO = [],
    isUserReviewed = false,
    mediaItem,
}: AnnotationActionsProviderProps) => {
    const projectId = useProjectIdentifier();
    const saveMutation = $api.useMutation(
        'post',
        '/api/projects/{project_id}/dataset/items/{dataset_item_id}/annotations'
    );

    const { data: project } = useProject();

    const [annotations, setAnnotations, undoRedoActions] = useUndoRedoState<Annotation[]>([]);

    useEffect(() => {
        const projectLabels = project?.task?.labels || [];

        const localAnnotations = mapServerAnnotationsToLocal(initialAnnotationsDTO, projectLabels);

        if (localAnnotations.length > 0) {
            undoRedoActions.reset(localAnnotations);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [initialAnnotationsDTO, project?.task?.labels]);

    const updateAnnotations = (updatedAnnotations: Annotation[]) => {
        const updatedMap = new Map(updatedAnnotations.map((ann) => [ann.id, ann]));

        setAnnotations((prevAnnotations) =>
            prevAnnotations.map((annotation) => updatedMap.get(annotation.id) ?? annotation)
        );
    };

    const addAnnotations = (shapes: Shape[], labels: Label[]) => {
        setAnnotations((prevAnnotations) => [
            ...prevAnnotations,
            ...shapes.map((shape) => ({
                shape,
                id: uuid(),
                labels,
            })),
        ]);
    };

    const deleteAnnotations = (annotationIds: string[]) => {
        setAnnotations((prevAnnotations) =>
            prevAnnotations.filter((annotation) => !annotationIds.includes(annotation.id))
        );
    };

    const submitAnnotations = async () => {
        const serverFormattedAnnotations = mapLocalAnnotationsToServer(annotations);

        await saveMutation.mutateAsync({
            params: { path: { dataset_item_id: mediaItem.id || '', project_id: projectId } },
            body: { annotations: serverFormattedAnnotations },
        });
    };

    return (
        <AnnotationsContext.Provider
            value={{
                isUserReviewed,
                annotations,

                // Local
                addAnnotations,
                updateAnnotations,
                deleteAnnotations,

                // Remote
                submitAnnotations,

                isSaving: saveMutation.isPending,
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
