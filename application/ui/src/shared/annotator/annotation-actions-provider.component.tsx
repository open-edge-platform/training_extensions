// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useEffect, useRef, useState } from 'react';

import { useProject } from 'hooks/api/project.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { get, isEmpty, isObject } from 'lodash-es';
import { $api } from 'src/api/client';
import type { components } from 'src/api/openapi-spec';
import type { DatasetItem, Label } from 'src/constants/shared-types';
import { v4 as uuid } from 'uuid';

import type { Annotation, Shape } from '../../features/annotator/types';

type ServerAnnotation = components['schemas']['DatasetItemAnnotation-Input'];

const mapServerAnnotationsToLocal = (serverAnnotations: ServerAnnotation[], projectLabels: Label[]): Annotation[] => {
    const labelMap = new Map(projectLabels.map((label) => [label.id, label]));

    return serverAnnotations.map((annotation) => {
        // We only get the ids of the labels
        const labels = annotation.labels
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
    addAnnotations: (shapes: Shape[]) => void;
    deleteAnnotations: (annotationIds: string[]) => void;
    updateAnnotations: (updatedAnnotations: Annotation[]) => void;
    submitAnnotations: () => Promise<void>;
    isUserReviewed: boolean;
    isSaving: boolean;
}

const AnnotationsContext = createContext<AnnotationsContextValue | null>(null);

type AnnotationActionsProviderProps = {
    children: ReactNode;
    mediaItem: DatasetItem;
};

const isUnannotatedError = (error: unknown): boolean => {
    return (
        isObject(error) && 'detail' in error && /Dataset item has not been annotated yet/i.test(String(error.detail))
    );
};

export const AnnotationActionsProvider = ({ children, mediaItem }: AnnotationActionsProviderProps) => {
    const projectId = useProjectIdentifier();
    const saveMutation = $api.useMutation(
        'post',
        '/api/projects/{project_id}/dataset/items/{dataset_item_id}/annotations'
    );

    const { data: serverAnnotations, error: fetchError } = $api.useQuery(
        'get',
        '/api/projects/{project_id}/dataset/items/{dataset_item_id}/annotations',
        {
            params: { path: { project_id: projectId, dataset_item_id: mediaItem.id || '' } },
        },
        {
            retry: (_failureCount, error: unknown) => !isUnannotatedError(error),
        }
    );

    const { data: project } = useProject();

    const [localAnnotations, setLocalAnnotations] = useState<Annotation[]>([]);
    const isDirty = useRef<boolean>(false);

    const updateAnnotations = (updatedAnnotations: Annotation[]) => {
        const updatedMap = new Map(updatedAnnotations.map((ann) => [ann.id, ann]));

        setLocalAnnotations((prevAnnotations) =>
            prevAnnotations.map((annotation) => updatedMap.get(annotation.id) ?? annotation)
        );
        isDirty.current = true;
    };

    const addAnnotations = (shapes: Shape[]) => {
        setLocalAnnotations((prevAnnotations) => [
            ...prevAnnotations,
            ...shapes.map((shape) => ({
                shape,
                id: uuid(),
                labels: [],
            })),
        ]);
        isDirty.current = true;
    };

    const deleteAnnotations = (annotationIds: string[]) => {
        setLocalAnnotations((prevAnnotations) =>
            prevAnnotations.filter((annotation) => !annotationIds.includes(annotation.id))
        );
        isDirty.current = true;
    };

    const submitAnnotations = async () => {
        if (!isDirty) return;

        const serverFormattedAnnotations = mapLocalAnnotationsToServer(localAnnotations);

        await saveMutation.mutateAsync({
            params: { path: { dataset_item_id: mediaItem.id || '', project_id: projectId } },
            body: { annotations: serverFormattedAnnotations },
        });

        isDirty.current = false;
    };

    useEffect(() => {
        if (!project || !serverAnnotations) return;

        const annotations = get(serverAnnotations, 'annotations', []);
        const projectLabels = project.task?.labels || [];

        if (annotations.length > 0) {
            const localFormattedAnnotations = mapServerAnnotationsToLocal(annotations, projectLabels);

            setLocalAnnotations(localFormattedAnnotations);
            isDirty.current = false;
        }
    }, [serverAnnotations, project]);

    useEffect(() => {
        if (!isEmpty(fetchError)) {
            setLocalAnnotations([]);
        }
    }, [fetchError]);

    return (
        <AnnotationsContext.Provider
            value={{
                isUserReviewed: get(serverAnnotations, 'user_reviewed', false),
                annotations: localAnnotations,

                // Local
                addAnnotations,
                updateAnnotations,
                deleteAnnotations,

                // Remote
                submitAnnotations,

                isSaving: saveMutation.isPending,
            }}
        >
            {children}
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
