// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useState } from 'react';

import { ToolType } from '../../components/tool-selection-bar/tools/interface';
import { useActiveTool } from './hooks/use-active-tool.hook';
import { Annotation } from './types';

type AnnotatorContext = {
    activeTool: ToolType | null;
    selectTool: (tool: ToolType) => void;

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
    const [annotations, setAnnotations] = useState<Annotation[]>(initialAnnotations);

    const handleSelectTool = (tool: ToolType) => {
        selectTool(tool);
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
