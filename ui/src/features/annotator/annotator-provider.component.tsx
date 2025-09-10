// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useState } from 'react';

import { ToolType } from '../../components/tool-selection-bar/tools/interface';
import { useActiveTool } from './hooks/use-active-tool.hook';
import { Annotation, MediaItem, RegionOfInterest } from './types';

type AnnotatorContext = {
    activeTool: ToolType | null;
    selectTool: (tool: ToolType) => void;

    updateAnnotation: (updatedAnnotation: Annotation) => void;

    roi: RegionOfInterest;
    annotations: Annotation[];
};

export const AnnotatorProviderContext = createContext<AnnotatorContext | null>(null);

export const AnnotatorProvider = ({ mediaItem, children }: { mediaItem: MediaItem; children: ReactNode }) => {
    const { activeTool, selectTool } = useActiveTool();
    const [annotations, setAnnotations] = useState<Annotation[]>(mediaItem.annotations);
    const roi = { x: 0, y: 0, width: mediaItem.width, height: mediaItem.height };

    const handleSelectTool = (tool: ToolType) => {
        selectTool(tool);
    };

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
                selectTool: handleSelectTool,

                updateAnnotation: handleUpdateAnnotation,

                annotations,

                roi,
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
