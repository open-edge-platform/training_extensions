// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { v4 as uuid } from 'uuid';

import { ToolType } from '../../components/tool-selection-bar/tools/interface';
import { useLoadImageQuery } from './hooks/use-load-image-query.hook';
import { Annotation, DatasetItem, RegionOfInterest, Shape } from './types';

type AnnotatorContext = {
    // Tools
    activeTool: ToolType | null;
    setActiveTool: Dispatch<SetStateAction<ToolType>>;

    // Annotations
    annotations: Annotation[];
    addAnnotation: (shape: Shape) => void;
    updateAnnotation: (updatedAnnotation: Annotation) => void;

    // Media item
    mediaItem: DatasetItem;
    image: ImageData;
    roi: RegionOfInterest;
};

export const AnnotatorProviderContext = createContext<AnnotatorContext | null>(null);

export const AnnotatorProvider = ({ mediaItem, children }: { mediaItem: DatasetItem; children: ReactNode }) => {
    const [activeTool, setActiveTool] = useState<ToolType>('selection');
    // todo: pass media annotations
    const [annotations, setAnnotations] = useState<Annotation[]>([]);

    const imageQuery = useLoadImageQuery(mediaItem);

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
                image: imageQuery.data,
                roi: { x: 0, y: 0, width: mediaItem.width, height: mediaItem.height },
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
