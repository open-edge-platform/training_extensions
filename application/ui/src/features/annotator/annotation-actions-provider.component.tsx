// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useEffect, useRef, useState } from 'react';

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { get } from 'lodash-es';
import { $api } from 'src/api/client';
import { components } from 'src/api/openapi-spec';
import { v4 as uuid } from 'uuid';

import { Annotation, DatasetItem, Label, Shape } from './types';

type ServerAnnotation = components['schemas']['DatasetItemAnnotation-Input'];
type LabelReference = components['schemas']['LabelReference'];
type ProjectLabel = components['schemas']['Label'];

const mapServerAnnotationsToLocal = (
    serverAnnotations: ServerAnnotation[],
    projectLabels: ProjectLabel[]
): Annotation[] => {
    return serverAnnotations.map((annotation) => {
        const labels: Label[] = annotation.labels
            .map((labelRef) => {
                const projectLabel = projectLabels.find((label) => label.id === labelRef.id);
                if (!projectLabel) return null;

                return {
                    id: projectLabel.id || uuid(),
                    name: projectLabel.name || 'Unknown',
                    color: projectLabel.color || '#888888',
                } as Label;
            })
            .filter((label): label is Label => label !== null);

        return {
            ...annotation,
            id: uuid(),
            labels,
        } as Annotation;
    });
};

const mapLocalAnnotationsToServer = (localAnnotations: Annotation[]): ServerAnnotation[] => {
    return localAnnotations.map((annotation) => ({
        labels: annotation.labels.map((label): LabelReference => ({ id: label.id })),
        shape: annotation.shape,
        ...(annotation.confidence !== undefined && { confidence: annotation.confidence }),
    }));
};

interface AnnotationsContextValue {
    annotations: Annotation[];
    addAnnotation: (shape: Shape) => void;
    deleteAnnotation: (annotationId: string) => void;
    updateAnnotation: (updatedAnnotation: Annotation) => void;
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

    const updateAnnotation = (updatedAnnotation: Annotation) => {
        const { id } = updatedAnnotation;

        setLocalAnnotations((prevAnnotations) =>
            prevAnnotations.map((annotation) => (annotation.id === id ? updatedAnnotation : annotation))
        );
        isDirty.current = true;
    };

    const addAnnotation = (shape: Shape) => {
        setLocalAnnotations((prevAnnotations) => [
            ...prevAnnotations,
            {
                shape,
                id: uuid(),
                labels: [{ id: uuid(), name: 'Default label', color: 'var(--annotation-fill)', isPrediction: false }],
            },
        ]);
        isDirty.current = true;
    };

    const deleteAnnotation = (annotationId: string) => {
        setLocalAnnotations((prevAnnotations) =>
            prevAnnotations.filter((annotation) => annotation.id !== annotationId)
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

        const projectLabels = project.task?.labels || [];
        const annotations = get(serverAnnotations, 'annotations', []);

        if (annotations.length > 0) {
            const localFormattedAnnotations = mapServerAnnotationsToLocal(annotations, projectLabels);

            setLocalAnnotations(localFormattedAnnotations);
            isDirty.current = false;
        }
    }, [serverAnnotations, project]);

    return (
        <AnnotationsContext.Provider
            value={{
                annotations: localAnnotations,

                // Local
                addAnnotation,
                updateAnnotation,
                deleteAnnotation,

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
