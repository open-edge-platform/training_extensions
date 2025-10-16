// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useEffect, useRef, useState } from 'react';

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { get } from 'lodash-es';
import { $api } from 'src/api/client';
import { components } from 'src/api/openapi-spec';
import { v4 as uuid } from 'uuid';

import { Annotation, DatasetItem, Shape } from './types';

type ServerAnnotation = components['schemas']['DatasetItemAnnotation-Input'];

const mapServerAnnotationsToLocal = (serverAnnotations: ServerAnnotation[]): Annotation[] => {
    return serverAnnotations.map((annotation) => {
        return {
            ...annotation,
            id: uuid(),
        } as Annotation;
    });
};

const mapLocalAnnotationsToServer = (localAnnotations: Annotation[]): ServerAnnotation[] => {
    return localAnnotations.map((annotation) => ({
        labels: annotation.labels,
        shape: annotation.shape,
        ...(annotation.confidence !== undefined && { confidence: annotation.confidence }),
    }));
};

interface AnnotationsContextValue {
    annotations: Annotation[];
    addAnnotations: (shapes: Shape[]) => void;
    deleteAnnotations: (annotationIds: string[]) => void;
    updateAnnotations: (updatedAnnotations: Annotation[]) => void;
    submitAnnotations: () => Promise<void>;
    isSaving: boolean;
}

const AnnotationsContext = createContext<AnnotationsContextValue | null>(null);

type AnnotationActionsProviderProps = {
    children: ReactNode;
    mediaItem: DatasetItem;
};

export const AnnotationActionsProvider = ({ children, mediaItem }: AnnotationActionsProviderProps) => {
    const projectId = useProjectIdentifier();
    const saveMutation = $api.useMutation(
        'post',
        '/api/projects/{project_id}/dataset/items/{dataset_item_id}/annotations'
    );
    const { data: serverAnnotations } = $api.useQuery(
        'get',
        '/api/projects/{project_id}/dataset/items/{dataset_item_id}/annotations',
        {
            params: { path: { project_id: projectId, dataset_item_id: mediaItem.id || '' } },
        }
    );
    const { data: project } = $api.useQuery('get', '/api/projects/{project_id}', {
        params: { path: { project_id: projectId } },
    });

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

        if (annotations.length > 0) {
            const localFormattedAnnotations = mapServerAnnotationsToLocal(annotations);

            setLocalAnnotations(localFormattedAnnotations);
            isDirty.current = false;
        }
    }, [serverAnnotations, project]);

    return (
        <AnnotationsContext.Provider
            value={{
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
