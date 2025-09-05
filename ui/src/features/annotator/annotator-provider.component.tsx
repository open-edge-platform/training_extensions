// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useState } from 'react';

import { ToolType } from '../../components/tool-selection-bar/tools/interface';
import { useActiveTool } from './hooks/use-active-tool.hook';
import { Annotation } from './types';

type AnnotatorContext = {
    activeTool: ToolType | null;
    selectTool: (tool: ToolType) => void;

    selectedAnnotation: Annotation | null;
    setSelectedAnnotation: (annotation: Annotation) => void;

    updateAnnotation: (updatedAnnotation: Annotation) => void;

    annotations: Annotation[];
};

export const AnnotatorProviderContext = createContext<AnnotatorContext | null>(null);

export const AnnotatorProvider = ({
    initialAnnotations,
    children,
}: {
    initialAnnotations: Annotation[];
    children: ReactNode;
}) => {
    const { activeTool, selectTool } = useActiveTool();
    const [selectedAnnotation, setSelectedAnnotation] = useState<Annotation | null>(null);
    const [annotations, setAnnotations] = useState<Annotation[]>(initialAnnotations);

    const handleSelectTool = (tool: ToolType) => {
        selectTool(tool);
    };

    const handleSelectAnnotation = (newAnnotation: Annotation) => {
        const { id } = newAnnotation;

        const ann = annotations.find((annotation) => annotation.id === id) || null;

        setSelectedAnnotation(ann);
    };

    const handleUpdateAnnotation = (updatedAnnotation: Annotation) => {
        const { id } = updatedAnnotation;

        setAnnotations((prevAnnotations) => [
            ...prevAnnotations.filter((annotation) => annotation.id !== id),
            updatedAnnotation,
        ]);
    };

    return (
        <AnnotatorProviderContext.Provider
            value={{
                activeTool,
                selectTool: handleSelectTool,

                selectedAnnotation,
                setSelectedAnnotation: handleSelectAnnotation,

                updateAnnotation: handleUpdateAnnotation,

                annotations,
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
