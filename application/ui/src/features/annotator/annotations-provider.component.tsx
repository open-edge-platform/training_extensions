// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useEffect, useState } from 'react';

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { $api } from 'src/api/client';
import { v4 as uuid } from 'uuid';

import { useAnnotator } from './annotator-provider.component';
import { Annotation, Shape } from './types';

interface AnnotationsContextValue {
    annotations: Annotation[];
    addAnnotation: (shape: Shape) => void;
    deleteAnnotation: (annotationId: string) => void;
    updateAnnotation: (updatedAnnotation: Annotation) => void;
    saveAnnotations: () => Promise<void>;
    isSaving: boolean;
}

const AnnotationsContext = createContext<AnnotationsContextValue | null>(null);

export const AnnotationsProvider = ({ children }: { children: ReactNode }): AnnotationsContextValue => {
    const { mediaItem } = useAnnotator();
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

    const [localAnnotations, setLocalAnnotations] = useState<Annotation[]>([]);
    const [isDirty, setIsDirty] = useState(false);

    const updateAnnotation = (updatedAnnotation: Annotation) => {
        const { id } = updatedAnnotation;

        setLocalAnnotations((prevAnnotations) =>
            prevAnnotations.map((annotation) => (annotation.id === id ? updatedAnnotation : annotation))
        );
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
    };

    const deleteAnnotation = (annotationId: string) => {
        setLocalAnnotations((prevAnnotations) =>
            prevAnnotations.filter((annotation) => annotation.id !== annotationId)
        );
    };

    const saveAnnotations = async () => {
        if (!isDirty) return;

        await saveMutation.mutateAsync({
            params: { path: { dataset_item_id: mediaItem.id || '', project_id: projectId } },
            body: { annotations: localAnnotations },
        });

        setIsDirty(false);
    };

    useEffect(() => {
        if (serverAnnotations) {
            setLocalAnnotations(serverAnnotations);
            setIsDirty(false);
        }
    }, [serverAnnotations]);

    return (
        <AnnotationsContext.Provider
            value={{
                annotations: localAnnotations,

                addAnnotation,
                updateAnnotation,
                deleteAnnotation,
                saveAnnotations,

                isSaving: saveMutation.isPending,
            }}
        >
            {children}
        </AnnotationsContext.Provider>
    );
};

export const useAnnotations = () => {
    const context = useContext(AnnotationsContext);

    if (context === null) {
        throw new Error('useAnnotations must be used within "AnnotationsProvider"');
    }

    return context;
};
