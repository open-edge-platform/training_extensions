// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { ToolType } from '../../components/tool-selection-bar/tools/interface';
import { Annotation, MediaItem } from './types';

type AnnotatorContext = {
    activeTool: ToolType | null;
    setActiveTool: Dispatch<SetStateAction<ToolType>>;

    updateAnnotation: (updatedAnnotation: Annotation) => void;

    mediaItem: MediaItem;
    annotations: Annotation[];
};

export const AnnotatorProviderContext = createContext<AnnotatorContext | null>(null);

export const AnnotatorProvider = ({ mediaItem, children }: { mediaItem: MediaItem; children: ReactNode }) => {
    const [activeTool, setActiveTool] = useState<ToolType>('selection');
    const [annotations, setAnnotations] = useState<Annotation[]>(mediaItem.annotations);

    const handleUpdateAnnotation = (updatedAnnotation: Annotation) => {
        const { id } = updatedAnnotation;

        setAnnotations((prevAnnotations) =>
            prevAnnotations.map((annotation) => (annotation.id === id ? updatedAnnotation : annotation))
        );
    };

    return (
        <AnnotatorProviderContext.Provider
            value={{
                activeTool,
                setActiveTool,

                updateAnnotation: handleUpdateAnnotation,

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
