// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { v4 as uuid } from 'uuid';

import { ToolType } from '../../components/tool-selection-bar/tools/interface';
import { Annotation, MediaItem, Shape } from './types';

type AnnotatorContext = {
    activeTool: ToolType | null;
    setActiveTool: Dispatch<SetStateAction<ToolType>>;

    addAnnotation: (shape: Shape) => void;
    updateAnnotation: (updatedAnnotation: Annotation) => void;

    mediaItem: MediaItem;
    annotations: Annotation[];
};

export const AnnotatorProviderContext = createContext<AnnotatorContext | null>(null);

export const AnnotatorProvider = ({ mediaItem, children }: { mediaItem: MediaItem; children: ReactNode }) => {
    const [activeTool, setActiveTool] = useState<ToolType>('selection');
    const [annotations, setAnnotations] = useState<Annotation[]>(mediaItem.annotations);

    const updateAnnotation = (updatedAnnotation: Annotation) => {
        const { id } = updatedAnnotation;

        setAnnotations((prevAnnotations) =>
            prevAnnotations.map((annotation) => (annotation.id === id ? updatedAnnotation : annotation))
        );
    };

    const addAnnotation = (shape: Shape) => {
        setAnnotations((prevAnnotations) => [
            ...prevAnnotations,
            {
                shape,
                id: uuid(),
                labels: [{ id: uuid(), name: 'Default label', color: 'var(--annotation-fill)', isPrediction: false }],
            },
        ]);
    };

    return (
        <AnnotatorProviderContext.Provider
            value={{
                activeTool,
                setActiveTool,

                addAnnotation,
                updateAnnotation,

                annotations,

                mediaItem,
            }}
        >
            {children}
        </AnnotatorProviderContext.Provider>
    );
};

export const useAnnotator = () => {
    const context = useContext(AnnotatorProviderContext);

    if (context === null) {
        throw new Error('useAnnotator was used outside of AnnotatorProvider');
    }

    return context;
};
